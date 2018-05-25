# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import torch

from functools import partial

from cfgs.config import cfg
from data_handler.detection_data_manager import FasterRCNNDataManager
from data_handler.data_manager_api import Mode
from loggers.tensorbord_logger import TensorBoardLogger
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_training_session import run_training_session
from model.feature_extractors.faster_rcnn_feature_extractors import create_feature_extractor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--config_dir', dest='config_dir', help='Path to config dir', type=str)
    args = parser.parse_args()

    # TODO: IB: cfg should be a local variable to enable h.p. sweeps like the following.
    # TODO: IB it can be assigned to the state of the trainer\evaluator\etc.
    global cfg
    cfg.load(args.config_dir)

    # possible_anchors_scales = [a,b,c]
    # for scale in possible_anchors_scales:
    #     cfg.scale = scale
    #     faster_rcnn = FasterRCNNTrainer(cfg)

    data_manager = FasterRCNNDataManager(mode=Mode.TRAIN, imdb_name=cfg.imdb_name,
                                        seed=cfg.RNG_SEED, num_workers=cfg.NUM_WORKERS, is_cuda=cfg.CUDA,
                                        batch_size=cfg.TRAIN.batch_size)

    train_logger = TensorBoardLogger(cfg.output_path)
    feature_extractors = create_feature_extractor(cfg.net, cfg.net_variant,
                                                  freeze=0, #TODO in cfg
                                                  pretrained_model_path=cfg.TRAIN.get("pretrained_model_path", None))
    model = FasterRCNNMetaArch(
                      feature_extractors,
                      class_names=data_manager.imdb.classes, # TODO: IB - data manager abstract should have get_classes function
                      is_class_agnostic=cfg.TRAIN.class_agnostic,
                      num_regression_outputs_per_bbox=4,
                      roi_pooler_name=cfg.POOLING_MODE)
    create_optimizer_fn = partial(torch.optim.SGD, momentum=cfg.TRAIN.MOMENTUM)
    run_training_session(data_manager, model, create_optimizer_fn, cfg, train_logger)

    # TODO: Think about the resume
    # if cfg.TRAIN.resume:
    #     load_name = cfg.get_ckpt_path()
    #     logger.info("loading checkpoint %s" % load_name)
    #     checkpoint = torch.load(load_name)
    #     cfg.session = checkpoint['session']
    #     cfg.start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr = optimizer.param_groups[0]['lr']
    #     if 'pooling_mode' in checkpoint.keys():
    #         cfg.POOLING_MODE = checkpoint['pooling_mode']
    #     logger.info("loaded checkpoint %s" % load_name)
