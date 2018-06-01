# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import numpy as np
import torch
from functools import partial

from data_handler.data_manager_api import Mode
from data_handler.detection_data_manager import FasterRCNNDataManager
from loggers.tensorbord_logger import TensorBoardLogger
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_training_session import run_training_session
from model.feature_extractors.faster_rcnn_feature_extractors import create_feature_extractor_from_ckpt
from util.config import ConfigProvider
from util.logging import set_root_logger


parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
parser.add_argument('--config_dir', dest='config_dir', help='Path to config dir', type=str)
args = parser.parse_args()

if not args.config_dir:
    raise Exception("Unable to run without config dir.")

cfg = ConfigProvider()
cfg.load(args.config_dir)
np.random.seed(cfg.RNG_SEED)

set_root_logger(cfg.get_log_path())
logger = logging.getLogger(__name__)

try:
    train_data_manager = FasterRCNNDataManager(
        mode=Mode.TRAIN, imdb_name=cfg.imdb_name,
        seed=cfg.RNG_SEED, num_workers=cfg.NUM_WORKERS, is_cuda=cfg.CUDA,
        cfg=cfg, batch_size=cfg.TRAIN.batch_size)

    train_logger = TensorBoardLogger(cfg.output_path)

    feature_extractors = create_feature_extractor_from_ckpt(
        cfg.net, cfg.net_variant, frozen_blocks=cfg.TRAIN.frozen_blocks,
        pretrained_model_path=cfg.TRAIN.get("pretrained_model_path", None))

    model = FasterRCNNMetaArch.create_with_random_normal_init(feature_extractors, cfg,
                                                              num_classes=train_data_manager.num_classes)

    create_optimizer_fn = partial(torch.optim.SGD, momentum=cfg.TRAIN.MOMENTUM)

    run_training_session(train_data_manager, model, create_optimizer_fn, cfg, train_logger, cfg.TRAIN.start_epoch)

except Exception:
    logger.error("Unexpected error: ", exc_info=True)
