"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly

This code is used for Feedback interactive network. Input channel is 6 (x, y, z, p_ch, n_ch, ot_model)
"""
import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
import torch.utils.data as Data
from torch import distributed as dist, multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, remap_lut_write, get_semantickitti_file_list
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings
import h5py
import random

#swt add
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

warnings.simplefilter(action='ignore', category=FutureWarning)

# tong ji yong de
def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True, area=5):
    ious_table = [f'{item:.2f}' for item in ious]
    header = ['method', 'Area', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.cfg_basename, str(area), f'{oa:.2f}', f'{macc:.2f}',
            f'{miou:.2f}'] + ious_table + [str(best_epoch), cfg.run_dir,
                                           wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def main(gpu, cfg): 
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME) # cfg.dataset.common.NAME how to get this? default.yaml
    
    # if cfg.rank == 0:
    #     Wandb.launch(cfg, cfg.wandb.use_wandb)
    #     writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    # else:
    writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # choose the network model
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank) # establish the model
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # Load data from my cfg.dataset.common.
    # train_loader, train2_loader, val_loader = load_data_cfg(cfg)
    cfg_Data = Datasets(cfg,ROOT_DIR)
    datasets = cfg_Data.get_Alldata() # format: [dataset1, dataset2,...]


       
    # seting the validate 
    # validate_fn = validate

    # optionally resume from a checkpoint
    model_module = model.module if hasattr(model, 'module') else model
    logging.info('Training from scratch')

    cfg.criterion_args.weight = None
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    # ===> start training
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    pred_list = None
    best_numClicks = 10
    for epoch in range(0, cfg.epochs):
        # if cfg.distributed:
        #     train_loader.sampler.set_epoch(epoch)
        if epoch == 0:
            datasets[0].strategy1All()
        else:
            datasets[0].strategy2(pred_list)

        train_loader = datasets[0].toTensor(cfg)

        pred_list, train_loss, train_miou, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg)

        is_best = False
        results = None
        for iters in range(0, 5):
            if iters == 0:
                datasets[1].strategy1Test()
            else:
                datasets[1].strategy2Test(results)
            
            val_loader = datasets[1].toTensor(cfg)
            results, val_miou, val_macc, val_oa, val_ious, val_accs = test(model, val_loader, cfg) # to validate  swt
            
            if val_miou is not None:
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Iters @E{iters},  val_oa:{val_oa:.2f} , val_miou: {val_miou:.2f}' )
        
        avg_click = np.mean(datasets[1].num_clicks)
        logging.info(
                        f'validation: avg_click: {avg_click:.2f}' )
        if avg_click < best_numClicks:
            best_numClicks = avg_click
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'Find a better minimum clicks ckpt @E{epoch}, avg_clicks {avg_click:.2f}, val_miou {val_miou:.2f}, val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
                    f'\n mious: {val_ious}')
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'minumum_clicks': avg_click},
                            is_best=is_best,
                            save_name = cfg.run_name + 'AVG_CLICK'
                            )
        
        if val_miou > best_val:
            is_best = True
            best_val = val_miou
            macc_when_best = val_macc
            oa_when_best = val_oa
            ious_when_best = val_ious
            best_epoch = epoch
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'Find a better mIOU ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
                    f'\nmious: {val_ious}')
               

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}, avg_clicks {avg_click:.2f}, minumum clicks{best_numClicks:.2f}')

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0: ## Strange？ why use rank
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
            is_best = False
    # After we initialize the model with train_loader, then we need to refine the model with train2_loader
    # In order to simplify the code, we split the whole project into three part, the next part: train with train2_loader is shown in 
    # mainFK_s2, in addition, we quit the random shuffle in Date_input

    

    if writer is not None:
        writer.close()


# has train_loader
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    pred_list = []
        
    for idx, data in pbar:
        points, labels = data

        points_split={}
        points_split['pos'] = points[:,:,0:3].cuda()
        
        # points_split['x']   = points[:,:,3:].transpose(1,2).cuda()  # wrong  correct
        points_split['x']   = points.transpose(1,2).cuda()
        labels = labels.cuda()


        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(points_split)
            loss = criterion(logits, labels)
            loss.backward()
        # optimize
            optimizer.step()
            optimizer.zero_grad()

        # update confusion matrix
        cm.update(logits.argmax(dim=1), labels)
        loss_meter.update(loss.item())

        # 25*1024
        pred = logits.argmax(dim=1)
        pred = pred.cpu()
        pred_list.append(pred)



        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    pred_list = np.concatenate(pred_list,axis=0)
    miou, macc, oa, ious, accs = cm.all_metrics()
    return pred_list, loss_meter.avg, miou, macc, oa, ious, accs



@torch.no_grad()
def test(model, val_loader, cfg, num_votes=1, data_transform=None):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    pre_list = []
    for idx, data in pbar:
        points, labels = data

        points_split={}
        points_split['pos'] = points[:,:,0:3].cuda()
        points_split['x']   = points.transpose(1,2).cuda()
        labels = labels.cuda()
        logits = model(points_split)
        cm.update(logits.argmax(dim=1), labels)
        # print(logits.shape)

        pred = np.argmax(logits.transpose(1,2).cpu(),2)
        pred = np.array(pred)
        pred = np.round(pred)
        pred = pred.astype(np.int)
        # print(pred.shape)
        pre_list.append(pred)
        # 
    pre_list = np.concatenate(pre_list,axis=0)
    # print(pre_list.shape) #(12454, 2048)
    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return pre_list, miou, macc, oa, ious, accs

    
    # return miou, macc, oa, ious, accs




class Datasets(object):
    def __init__(self, cfg, ROOT_DIR = None):
        self.path = cfg.dataset.common.data_root
        self.num_point = cfg.dataset.common.numpoints
        self.click_flag = cfg.dataset.common.click_flag2train
        self.click_flag2 = cfg.dataset.common.click_flag2test
        self.channels = cfg.model.encoder_args.in_channels - cfg.click_channel
        self.split_ratio = np.array([float(x) for x in cfg.dataset.common.split_ratio.split(',')])
        self.cls_group1 = np.array([int(x) for x in cfg.cls_group1.split(',')])
        self.cls_group2 = np.array([int(x) for x in cfg.cls_group2.split(',')])
        self.test_species = cfg.test_species
        self.root_dir = ROOT_DIR

    def load_data(self):
        H5names = os.listdir(os.path.join(self.root_dir, '../', self.path))
        for i in range(len(H5names)):
            H5name = H5names[i]
            path2 = os.path.join(self.root_dir, '../', self.path, H5name)
            f = h5py.File(path2, 'r')
            points_h5 = f['data'][:]
            label_h5 = f['label'][:]
            label2_h5 = f['label2'][:]
            if i == 0:
                points = points_h5
                label = label_h5
                label2 = label2_h5
            else:
                points = np.concatenate((points, points_h5), axis=0)
                label = np.concatenate((label, label_h5), axis=0)
                label2 = np.concatenate((label2, label2_h5), axis=0)

        points = points[:, :, 0:self.channels]


        label = np.expand_dims(label, axis=-1)
        block = np.concatenate((points, label), axis=2)
        block = self.sample(block, self.num_point)  # 4096

        points = block[:, :, 0:self.channels]
        label = block[:, :, self.channels]
        label2 = label2[:, 0]

        dataset = Datastruct(points, label, label2)
        return dataset

    def __classify2species(self, points, label, label2, cls_group):
        # -------------- Divide training according to the species ---------
        # type_to_id = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4, 'Person_sitting': 5,
        #               'Cyclist': 6, 'Tram': 7, 'Misc': 8}

        cond = np.isin(label2, cls_group, invert=False)
        new_points = points[cond, ...]
        new_label = label[cond, ...]
        new_label2 = label2[cond, ...]
        return new_points, new_label, new_label2

    def __data_split(self, pr_dataset):
        # --------------Divide dataset into several parts: train, val, or test--------
        datasets = []
        tr_num = 0

        points = pr_dataset.points
        label = pr_dataset.label
        label2 = pr_dataset.label2
        num_blocks = points.shape[0]

        for i in range(len(self.split_ratio)):
            one_points = points[tr_num:tr_num + int(num_blocks * self.split_ratio[i]), ...]
            one_label = label[tr_num:tr_num + int(num_blocks * self.split_ratio[i]), ...]
            one_label2 = label2[tr_num:tr_num + int(num_blocks * self.split_ratio[i]), ...]
            dataset = Datastruct(one_points, one_label, one_label2)
            datasets.append(dataset)
            tr_num += int(num_blocks * self.split_ratio[i])

        return datasets


    def sample(self, one_shot, num_point=1024):
        """This code is to up or down sample points to fit the NUMBER of points
        INPUT
            one_shot:  B*N*C
            num_points: 1024/2048/4096/8192"""
        l = one_shot.shape[1]
        idx = np.arange(l)
        if l >= num_point:  # down sample
            np.random.shuffle(idx)
            return one_shot[:, idx[0:num_point], ...]
        else:
            batch = num_point // l
            els = num_point % l
            one_shot_rep = np.tile(one_shot, (1, batch, 1))
            return np.concatenate((one_shot_rep, one_shot[:, idx[0:els], :]), axis=1)

    def get_Alldata(self):
        datasets = self.__data_split(self.load_data())
        return datasets

class Datastruct(object):
    def __init__(self, points, label, label2):
        self.points = points - np.mean(points, axis=1).reshape((points.shape[0],1,points.shape[2]))
        self.points = self.points - np.mean(self.points, axis=1).reshape((points.shape[0],1,points.shape[2]))  # It is necessary ??
        self.label = label
        self.label2 = label2
        self.masks = None
        self.pchs = None
        self.nchs = None
        self.num_clicks = None

        # some parameters
        self.sigma = 0.1
        self.pr = 0.3
        self.s1num = 2  # strategy1: init number of clicks
        self.num_points = points.shape[1]
        self.num_blocks = points.shape[0]
        self.thr = 0.95

    def toTensor(self, cfg):
        """
        OUTPUT:
        """
        blocks = np.concatenate((self.points,
                                 np.expand_dims(self.pchs, axis=-1),
                                 np.expand_dims(self.nchs, axis=-1),
                                 np.expand_dims(self.masks, axis=-1)), axis=-1)

        blocks = torch.tensor(blocks, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        TensorPoints = Data.TensorDataset(blocks, label)

        loader = torch.utils.data.DataLoader(TensorPoints,
                                             batch_size=cfg.batch_size,
                                             num_workers=0,
                                             shuffle=False,
                                             pin_memory=False)
        return loader

    def strategy1All(self):
        """
        Strategy1 to initialize the clicks for training
        P:1-self.s1num clicks
        N: 0-self.s1num clicks
        """
        num = self.points.shape[0]
        new_blocks = []
        n_c = []
        for i in range(num):
            block = self.points[i, ...]
            la = self.label[i, ...]
            cond = la == 1
            obj = block[cond, 0:3]


            P_num = random.randint(1, self.s1num)
            positive_clicks = random.sample(range(obj.shape[0]), P_num)
            positive_clicks = obj[positive_clicks, :]
            pch = np.zeros((block.shape[0], 1), dtype=np.float)
            for j in range(P_num):
                dist = np.linalg.norm(block - positive_clicks[j, ...], ord=2, axis=-1, keepdims=True)
                pch = pch + np.exp(-np.square(dist) / (2 * self.sigma))

            N_num = random.randint(0, self.s1num)
            nch = np.zeros((block.shape[0], 1), dtype=np.float)
            cond = la == 0
            if np.sum(cond) > N_num and N_num > 0:
                bg = block[cond, 0:3]
                negative_clicks = random.sample(range(bg.shape[0]), N_num)
                negative_clicks = bg[negative_clicks, :]
                for j in range(N_num):
                    dist = np.linalg.norm(block - negative_clicks[j, ...], ord=2, axis=-1, keepdims=True)
                    nch = nch + np.exp(-np.square(dist) / (2 * self.sigma))
            else:
                N_num = 0

            mask = np.zeros((block.shape[0], 1), dtype=np.float)
            newB = np.concatenate((block, pch, nch, mask), axis=1)
            newB = np.expand_dims(newB, axis=0)
            new_blocks.append(newB)
            n_c.append(P_num + N_num)
            # print('finish:%d'%(i))
        new_blocks = np.concatenate(new_blocks, axis=0)
        self.pchs = new_blocks[:, :, -3]
        self.nchs = new_blocks[:, :, -2]
        self.masks = new_blocks[:, :, -1]
        self.num_clicks = np.array(n_c, dtype=np.int)
        return 0

    def __strategy1Part(self, idx):
        """
        For one sample to initiate with strategy1
        :param idx:
        :return:
        """
        new_blocks = []
        n_c = []
        block = self.points[idx, ...]
        la = self.label[idx, ...]
        cond = la == 1
        obj = block[cond, 0:3]

        P_num = random.randint(1, self.s1num)
        positive_clicks = random.sample(range(obj.shape[0]), P_num)
        positive_clicks = obj[positive_clicks, :]
        pch = np.zeros((block.shape[0], 1), dtype=np.float)
        for j in range(P_num):
            dist = np.linalg.norm(block - positive_clicks[j, ...], ord=2, axis=-1, keepdims=True)
            pch = pch + np.exp(-np.square(dist) / (2 * self.sigma))

        N_num = random.randint(0, self.s1num)
        nch = np.zeros((block.shape[0], 1), dtype=np.float)
        cond = la == 0
        if np.sum(cond) > N_num and N_num > 0:
            bg = block[cond, 0:3]
            negative_clicks = random.sample(range(bg.shape[0]), N_num)
            negative_clicks = bg[negative_clicks, :]
            for j in range(N_num):
                dist = np.linalg.norm(block - negative_clicks[j, ...], ord=2, axis=-1, keepdims=True)
                nch = nch + np.exp(-np.square(dist) / (2 * self.sigma))
        else:
            N_num = 0

        self.num_clicks[idx] = P_num + N_num
        self.pchs[idx, ...] = pch.reshape(self.num_points)
        self.nchs[idx, ...] = nch.reshape(self.num_points)
        return 0

    def strategy2(self, pred_list):
        """
        This is to use the strategy2 while training
        :param pred_list: prediction last iteration
        :return:
        """
        for i in range(self.points.shape[0]):
            pred_one = pred_list[i, ...]  # 4096,1
            points_one = self.points[i, ...]  # just for KITTI, Apollo, Scannet
            la_one = self.label[i, ...]

            # Pr to initiate the original clicks
            rdm = random.random()
            if self.pr > rdm:
                self.__strategy1Part(i)
            rdm = random.random()
            if self.pr > rdm:
                self.masks[i, ...] = np.zeros(self.num_points, dtype=np.float)
            else:
                self.masks[i, ...] = pred_one

            # IOU calculate for OBJ
            gt1 = np.sum(np.multiply(pred_one, la_one))
            gf1 = np.sum(pred_one) + np.sum(la_one) - gt1
            IOU1 = gt1 / float(gf1)
            # IOU calculate for BG
            gt2 = np.sum(np.multiply(-pred_one + 1, -la_one + 1))  # ????
            gf2 = np.sum(-pred_one + 1) + np.sum(-la_one + 1) - gt2
            # IOU2 = gt2 / float(gf2)
            # mIoU = (IOU1+IOU2)/2

            if IOU1 <= self.thr:  # IoU < threshold: start next iteration
                area1 = np.multiply(-pred_one + 1, la_one)  # the area belongs to object, but not segment successfully
                area2 = np.multiply(pred_one, -la_one + 1)  # the area belongs to BG, but wrongly classify it as object
                self.num_clicks[i] += 1

                if np.sum(area1) > np.sum(area2):
                    cond = area1 == 1
                    obj = points_one[cond, ...]

                    cent = np.random.randint(obj.shape[0], size=1)
                    cent = obj[cent]
                    dist = np.linalg.norm(points_one - cent, ord=2, axis=-1)
                    self.pchs[i, ...] += np.exp(-np.square(dist) / (2 * self.sigma))
                else:
                    cond = area2 == 1
                    obj = points_one[cond, ...]
                    cent = np.random.randint(obj.shape[0], size=1)
                    cent = obj[cent]
                    dist = np.linalg.norm(points_one - cent, ord=2, axis=-1)
                    self.nchs[i, ...] += np.exp(-np.square(dist) / (2 * self.sigma))

        return 0

    def strategy1Test(self):
        """
        This is to simulat the strategy1 while testing
        :return:
        """
        num = self.points.shape[0]
        new_blocks = []
        n_c = []
        for i in range(num):
            block = self.points[i, ...]
            la = self.label[i, ...]
            cond = la == 1
            obj = block[cond, 0:3]

            P_num = 1  # random.randint(1, self.s1num)
            positive_clicks = random.sample(range(obj.shape[0]), P_num)
            positive_clicks = obj[positive_clicks, :]
            pch = np.zeros((block.shape[0], 1), dtype=np.float)
            for j in range(P_num):
                dist = np.linalg.norm(block - positive_clicks[j, ...], ord=2, axis=-1, keepdims=True)
                pch = pch + np.exp(-np.square(dist) / (2 * self.sigma))

            nch = np.zeros((block.shape[0], 1), dtype=np.float)
            mask = np.zeros((block.shape[0], 1), dtype=np.float)
            newB = np.concatenate((block, pch, nch, mask), axis=1)
            newB = np.expand_dims(newB, axis=0)
            new_blocks.append(newB)
            n_c.append(P_num)
            # print('finish:%d'%(i))
        new_blocks = np.concatenate(new_blocks, axis=0)
        self.pchs = new_blocks[:, :, -3]
        self.nchs = new_blocks[:, :, -2]
        self.masks = new_blocks[:, :, -1]
        self.num_clicks = np.array(n_c, dtype=np.int)
        return 0

    def strategy2Test(self, pred_list):
        """
        This is to simulate the click strategy2 during testing

        :param pred_list: prediction last iteration
        :return:
        """
        for i in range(self.points.shape[0]):
            pred_one = pred_list[i, ...]  # 4096,1
            points_one = self.points[i, ...]  # just for KITTI, Apollo, Scannet
            la_one = self.label[i, ...]

            # IOU calculate for OBJ
            gt1 = np.sum(np.multiply(pred_one, la_one))
            gf1 = np.sum(pred_one) + np.sum(la_one) - gt1
            IOU1 = gt1 / float(gf1)
            # IOU calculate for BG
            gt2 = np.sum(np.multiply(-pred_one + 1, -la_one + 1))  # ????
            gf2 = np.sum(-pred_one + 1) + np.sum(-la_one + 1) - gt2
            # IOU2 = gt2 / float(gf2)
            # mIoU = (IOU1+IOU2)/2

            if (IOU1 <= self.thr):  # IoU > threshold --> stop clicking, else start the next iteration
                area1 = np.multiply(-pred_one + 1, la_one)  # the area belongs to object, but not segment successfully
                area2 = np.multiply(pred_one, -la_one + 1)  # the area belongs to BG, but wrongly classify it as object
                self.num_clicks[i] += 1
                self.masks[i, ...] = pred_one

                if (np.sum(area1) > np.sum(area2)):
                    cond = area1 == 1
                    obj = points_one[cond, ...]

                    cent = np.random.randint(obj.shape[0], size=1)
                    cent = obj[cent]

                    dist = np.linalg.norm(points_one - cent, ord=2, axis=-1)
                    self.pchs[i, ...] += np.exp(-np.square(dist) / (2 * self.sigma))
                else:
                    cond = area2 == 1
                    obj = points_one[cond, ...]
                    cent = np.random.randint(obj.shape[0], size=1)
                    cent = obj[cent]
                    dist = np.linalg.norm(points_one - cent, ord=2, axis=-1)
                    self.nchs[i, ...] += np.exp(-np.square(dist) / (2 * self.sigma))

        return 0







if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name) # eg ../cfgs/scannet/
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']  #judge if to train, True of False
    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path) # get the directory which stores the model(checkpoint)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))# to generate the directory to store the model checkpoint
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")  # while running, sotre the yaml fils 
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)



# CUDA_VISIBLE_DEVICES=0 python examples/interactivesg4/mainFK.py --cfg cfgs/interactivesg4/pointnext-l-trainScan.yaml
# CUDA_VISIBLE_DEVICES=0 python examples/interactivesg4/mainFK.py --cfg cfgs/interactivesg4/pointnext-l-trainApo.yaml

# CUDA_VISBLE_DEVICES=0 python examples/interactivesg4/mainFK.py --cfg cfgs/interactivesg4/TiisNet-trainApo.yaml
# CUDA_VISBLE_DEVICES=0 python examples/interactivesg4/mainFK.py --cfg cfgs/interactivesg4/pointmetabase-l-trainApo.yaml


