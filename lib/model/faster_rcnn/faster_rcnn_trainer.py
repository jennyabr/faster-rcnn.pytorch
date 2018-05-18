import time
import logging

import torch
import torch.nn as nn

from model.feature_extractors.feature_extractors_factory import FeatureExtractorsFactory
from model.utils.net_utils import adjust_learning_rate, clip_gradient
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch

from cfgs.config import cfg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FasterRCNNTrainer(object):
    def __init__(self, data_manager, train_logger):
        self. train_logger = train_logger
        self.data_manager = data_manager
        feature_extractors = FeatureExtractorsFactory(cfg.net, cfg.TRAIN.pretrained_model_path)
        self.faster_rcnn = FasterRCNNMetaArch(
                              feature_extractors,
                              class_names=self.data_manager.imdb.classes,
                              predict_bbox_per_class=cfg.TRAIN.class_agnostic,  # TODO is it in TRAIN?
                              num_regression_outputs_per_bbox=4,  # TODO parametrize
                              roi_pooler_name=cfg.POOLING_MODE)

    def train_model(self, optimizer):
        '''

        :param optimizer: default optimizer is sgd
        :return:
        '''
        #TODO: IB - create sub functions
        lr = cfg.TRAIN.LEARNING_RATE

        def get_trainable_params():
            trainable_params = []
            for key, value in dict(self.faster_rcnn.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        trainable_params += [{'params': [value],
                                    'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        trainable_params += [{'params': [value],
                                    'lr': lr,
                                    'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            return trainable_params
        trainable_params = get_trainable_params()

        if not optimizer:
            optimizer = torch.optim.SGD(trainable_params, momentum=cfg.TRAIN.MOMENTUM)

        if cfg.TRAIN.resume:
            load_name = cfg.get_ckpt_path()
            logger.info("loading checkpoint %s" % load_name)
            checkpoint = torch.load(load_name)
            cfg.session = checkpoint['session']
            cfg.start_epoch = checkpoint['epoch']
            self.faster_rcnn.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
            logger.info("loaded checkpoint %s" % load_name)

        if cfg.mGPUs:
            self.faster_rcnn = nn.DataParallel(self.faster_rcnn)

        if cfg.CUDA:
            self.faster_rcnn.cuda()

        iters_per_epoch = int(self.data_manager.train_size / cfg.TRAIN.batch_size)

        for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.max_epochs + 1):
            self.faster_rcnn.train()  # setting to train mode TODO why each epoch?
            loss_temp = 0
            start = time.time()

            if epoch % (cfg.TRAIN.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, cfg.TRAIN.GAMMA)
                lr *= cfg.TRAIN.GAMMA

            for step in range(iters_per_epoch):
                im_data, im_info, gt_boxes, num_boxes = next(self.data_manager)

                self.faster_rcnn.zero_grad()
                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, rois_label = self.faster_rcnn(im_data, im_info, gt_boxes, num_boxes)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                loss_temp += loss.data[0]

                # backward
                optimizer.zero_grad()
                loss.backward()

                if cfg.net == "vgg16":  # TODO why?
                    clip_gradient(self.faster_rcnn, 10.)

                optimizer.step()
                if step % cfg.TRAIN.disp_interval == 0:  # TODO add: or (step + 1) == iters_per_epoch + update the loss aproprietly
                    end = time.time()
                    if step > 0:
                        loss_temp /= cfg.TRAIN.disp_interval  # TODO understand

                    def collect_data():
                        # TODO: IB - we are saving metrics for a *single* batch every 100\500 batches and that creates a
                        # TODO: IB - very noisy measurement. The metrics that should be saved are an average
                        # TODO: IB - of the last 100\500 batches (it's preferable if possible to aggregate this average
                        # TODO: IB - on the gpu to avoid cpu-gpu overhead every batch)
                        #
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
                        return loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, fg_cnt, bg_cnt
                    loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, fg_cnt, bg_cnt = collect_data()
                    logger.info("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e \n"                                
                                "\t\t\tfg/bg=(%d/%d), time cost: %f \n" 
                                "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f"
                                % (cfg.TRAIN.session, epoch, step, iters_per_epoch, loss_temp, lr,
                                   fg_cnt, bg_cnt, end - start,
                                   loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                    def log_summary():
                        total_step = epoch * iters_per_epoch + step
                        self.train_logger.scalar_summary('loss', loss_temp, total_step)
                        self.train_logger.scalar_summary('loss_rpn_cls', loss_rpn_cls, total_step)
                        self.train_logger.scalar_summary('loss_rpn_box', loss_rpn_box, total_step)
                        self.train_logger.scalar_summary('loss_rcnn_cls', loss_rcnn_cls, total_step)
                        self.train_logger.scalar_summary('loss_rcnn_box', loss_rcnn_box, total_step)
                    log_summary()

                    loss_temp = 0
                    start = time.time()

            end1 = time.time()

            def save_checkpoint():
                if cfg.mGPUs:
                    ckpt_model = self.faster_rcnn.module.state_dict()
                else:
                    ckpt_model = self.faster_rcnn.state_dict()
                save_to = cfg.get_ckpt_path(epoch)
                logger.info('Saving model checkpoint to {}'.format(save_to))
                torch.save({'session': cfg.TRAIN.session,
                             'epoch': epoch + 1,
                             'model': ckpt_model,
                             'optimizer': optimizer.state_dict(),
                             'pooling_mode': cfg.POOLING_MODE,
                             'class_agnostic': cfg.TRAIN.class_agnostic}, save_to)
            save_checkpoint()

            end2 = time.time()
            logger.info("Finished epoch {} in {} ms (including checkpoint saving {} ms)".format(epoch, end1 - start, end2 - start))
