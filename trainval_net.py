# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from functools import partial

from cfgs.config import cfg
from data_handler.detection_data_manager import FasterRCNNDataManager
from data_handler.data_manager_api import Mode
from loggers.tensorbord_logger import TensorBoardLogger
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_training_session import run_training_session
from model.feature_extractors.faster_rcnn_feature_extractors import create_feature_extractor_from_ckpt
from cfgs.config import get_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--config_dir', dest='config_dir', help='Path to config dir', type=str,
                        default='/home/jenny/gripper2/test_on_p100/cfgs/') #TODO: IB - delete the default
    args = parser.parse_args()

    if not args.config_dir:
        raise Exception("Unable to run without config dir.")

    # TODO: IB: cfg should be a local variable to enable h.p. sweeps like the following.
    # TODO: IB it can be assigned to the state of the trainer\evaluator\etc.
    global cfg
    cfg.load(args.config_dir)
    experiment_name = 'faster_rcnn_vgg_voc'

    # possible_anchors_scales = [a,b,c]
    # for scale in possible_anchors_scales:
    #     cfg.scale = scale
    #     faster_rcnn = FasterRCNNTrainer(cfg)

    logger = get_logger(__name__)

    train_data_manager = FasterRCNNDataManager(
        mode=Mode.TRAIN, imdb_name=cfg.imdb_name, seed=cfg.RNG_SEED, num_workers=cfg.NUM_WORKERS, is_cuda=cfg.CUDA,
        batch_size=cfg.TRAIN.batch_size)
    train_logger = TensorBoardLogger(cfg.output_path)
    feature_extractors = create_feature_extractor_from_ckpt(
        cfg.net, cfg.net_variant, frozen_blocks=cfg.TRAIN.frozen_blocks,
        pretrained_model_path=cfg.TRAIN.get("pretrained_model_path", None))
    model = FasterRCNNMetaArch.create_with_random_normal_init(feature_extractors, cfg,
                                                              num_classes=train_data_manager.num_classes)
    create_optimizer_fn = partial(torch.optim.SGD, momentum=cfg.TRAIN.MOMENTUM)
    run_training_session(train_data_manager, model, create_optimizer_fn, cfg, train_logger, cfg.TRAIN.start_epoch)

    # eval_data_manager = FasterRCNNDataManager(mode=Mode.EVAL, imdb_name=cfg.imdbval_name,
    #                                           seed=cfg.RNG_SEED, num_workers=cfg.NUM_WORKERS, is_cuda=cfg.CUDA,
    #                                           batch_size=cfg.TRAIN.batch_size)
