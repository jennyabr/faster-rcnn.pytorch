# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import os
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import Sampler

from model.feature_extractors.resnet_for_faster_rcnn import ResNetForFasterRCNN
from model.feature_extractors.vgg16_for_faster_rcnn import VGG16ForFasterRCNN
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from cfgs.config import cfg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BDSampler(Sampler):
    def __init__(self, train_size, batch_size, seed):
        self.seed = seed #TODO is it done no the CPU?
        torch.manual_seed(seed)

        self.data_size = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.data_size  # TODO (self.data_size + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--config_dir', dest='config_dir', help='Path to config dir', type=str)
    args = parser.parse_args()
    global cfg
    cfg.load(args.config_dir)

    if cfg.TRAIN.use_tfboard:
        from model.utils.logger import Logger
        tensorboard = Logger(cfg.output_path)
    else:
        tensorboard = None

    print(cfg.DATA_DIR)

    imdb, roidb, ratio_list, ratio_index = combined_roidb(cfg.imdb_name)
    train_size = len(roidb)

    logger.info('{:d} roidb entries'.format(len(roidb)))

    sampler_batch = BDSampler(train_size, cfg.TRAIN.batch_size, cfg.RNG_SEED)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, cfg.TRAIN.batch_size,
                             imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.batch_size,
                                             sampler=sampler_batch, num_workers=cfg.num_workers)

    # initilize the tensor holder
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    if cfg.CUDA:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)


    if cfg.net == 'vgg16':
        model_path = os.path.join(cfg.DATA_DIR, 'pretrained_model/vgg16_caffe.pth')
        feature_extractors = VGG16ForFasterRCNN(pretrained=True, model_path=model_path)
    elif cfg.net == 'res101':
        model_path = os.path.join(cfg.DATA_DIR, 'pretrained_model/resnet101_caffe.pth')
        feature_extractors = ResNetForFasterRCNN(pretrained=True, model_path=model_path)
    else:
        feature_extractors = None
        raise Exception("network is not defined")

    faster_rcnn = FasterRCNNMetaArch(feature_extractors,
                                     class_names=imdb.classes,
                                     predict_bbox_per_class=cfg.TRAIN.class_agnostic,
                                     num_regression_outputs_per_bbox=4,
                                     roi_pooler_name='align')

    lr = cfg.TRAIN.LEARNING_RATE

    params = []
    for key, value in dict(faster_rcnn.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value],
                            'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value],
                            'lr': lr,
                            'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if cfg.TRAIN.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif cfg.TRAIN.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    else:
        optimizer = None
        raise AssertionError('optimizer not defined')

    if cfg.TRAIN.resume:
        load_name = cfg.get_ckpt_path()
        logger.info("loading checkpoint %s" % load_name)
        checkpoint = torch.load(load_name)
        cfg.session = checkpoint['session']
        cfg.start_epoch = checkpoint['epoch']
        faster_rcnn.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        logger.info("loaded checkpoint %s" % load_name)

    if cfg.mGPUs:
        faster_rcnn = nn.DataParallel(faster_rcnn)

    if cfg.CUDA:
        faster_rcnn.cuda()

    iters_per_epoch = int(train_size / cfg.TRAIN.batch_size)

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.max_epochs + 1):
        # setting to train mode
        faster_rcnn.train()
        loss_temp = 0
        start = time.time()

        if epoch % (cfg.TRAIN.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, cfg.TRAIN.GAMMA)
            lr *= cfg.TRAIN.GAMMA

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            faster_rcnn.zero_grad()
            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, rois_label = \
                faster_rcnn(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()
            loss.backward()

            if cfg.net == "vgg16":
                clip_gradient(faster_rcnn, 10.)

            optimizer.step()
            if step % cfg.TRAIN.disp_interval == 0: #TODO add: or (step + 1) == iters_per_epoch + update the loss aproprietly
                end = time.time()
                if step > 0:
                    loss_temp /= cfg.TRAIN.disp_interval #TODO understand

                if cfg.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_box.mean().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.data[0]
                    loss_rpn_box = rpn_loss_box.data[0]
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                logger.info("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                      % (cfg.TRAIN.session, epoch, step, iters_per_epoch, loss_temp, lr))
                logger.info("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                logger.info("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f"
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                if cfg.TRAIN.use_tfboard:
                    total_step = epoch * iters_per_epoch + step
                    tensorboard.scalar_summary('loss',          loss_temp,     total_step)
                    tensorboard.scalar_summary('loss_rpn_cls',  loss_rpn_cls,  total_step)
                    tensorboard.scalar_summary('loss_rpn_box',  loss_rpn_box,  total_step)
                    tensorboard.scalar_summary('loss_rcnn_cls', loss_rcnn_cls, total_step)
                    tensorboard.scalar_summary('loss_rcnn_box', loss_rcnn_box, total_step)

                loss_temp = 0
                start = time.time()

        if cfg.mGPUs:
            ckpt_model = faster_rcnn.module.state_dict()
        else:
            ckpt_model = faster_rcnn.state_dict()
        save_to = cfg.get_ckpt_path(epoch)
        save_checkpoint({'session': cfg.TRAIN.session,
                         'epoch': epoch + 1,
                         'model': ckpt_model,
                         'optimizer': optimizer.state_dict(),
                         'pooling_mode': cfg.POOLING_MODE,
                         'class_agnostic': cfg.TRAIN.class_agnostic}, save_to)
        logger.info('save model: {}'.format(save_to))

        end = time.time()
        logger.info(end - start)
