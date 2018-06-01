# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
import torch
from functools import partial

from data_handler.data_manager_api import Mode
from data_handler.detection_data_manager import FasterRCNNDataManager
from loggers.tensorbord_logger import TensorBoardLogger
from model.faster_rcnn.faster_rcnn_evaluation import faster_rcnn_evaluation
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_training_session import run_training_session
from model.faster_rcnn.faster_rcnn_postprocessing import faster_rcnn_postprocessing
from model.faster_rcnn.faster_rcnn_prediction import faster_rcnn_prediction
from model.faster_rcnn.faster_rcnn_visualization import faster_rcnn_visualization
from model.feature_extractors.faster_rcnn_feature_extractors import create_feature_extractor_from_ckpt
from util.config import ConfigProvider
from util.logging import set_root_logger


config_file = '/home/jenny/gripper2/test_on_p100/cfgs/resnet101.yml'

cfg = ConfigProvider()
cfg.load(config_file)
cfg.GPU_ID = 0
np.random.seed(cfg.RNG_SEED)

set_root_logger(cfg.get_log_path())
logger = logging.getLogger(__name__)


def create_and_train():
    train_data_manager = FasterRCNNDataManager(
        mode=Mode.TRAIN, imdb_name=cfg.imdb_name, num_workers=cfg.NUM_WORKERS, is_cuda=cfg.CUDA, cfg=cfg,
        batch_size=cfg.TRAIN.batch_size)

    train_logger = TensorBoardLogger(cfg.output_path)

    feature_extractors = create_feature_extractor_from_ckpt(
        cfg.net, cfg.net_variant, frozen_blocks=cfg.TRAIN.frozen_blocks,
        pretrained_model_path=cfg.TRAIN.get("pretrained_model_path", None))

    model = FasterRCNNMetaArch.create_with_random_normal_init(feature_extractors, cfg,
                                                              num_classes=train_data_manager.num_classes)

    create_optimizer_fn = partial(torch.optim.SGD, momentum=cfg.TRAIN.MOMENTUM)

    run_training_session(train_data_manager, model, create_optimizer_fn, cfg, train_logger, cfg.TRAIN.start_epoch)


def pred_eval(predict_on_epoch):
    ckpt_path = cfg.get_last_ckpt_path()
    model = FasterRCNNMetaArch.create_from_ckpt(ckpt_path)
    model.cuda()
    data_manager = FasterRCNNDataManager(mode=Mode.INFER,
                                         imdb_name=cfg.imdbval_name,
                                         num_workers=cfg.NUM_WORKERS,
                                         is_cuda=cfg.CUDA,
                                         batch_size=cfg.TRAIN.batch_size,
                                         cfg=cfg)

    faster_rcnn_prediction(data_manager, model, cfg, predict_on_epoch)

    faster_rcnn_postprocessing(data_manager, model, cfg, predict_on_epoch)

    detections_path = cfg.get_postprocessed_detections_path(predict_on_epoch)
    eval_path = cfg.get_evals_dir_path(predict_on_epoch)
    faster_rcnn_evaluation(data_manager, cfg, detections_path, eval_path)

    faster_rcnn_visualization(data_manager, cfg, predict_on_epoch)


try:
    create_and_train()
    pred_eval(predict_on_epoch=7)
except Exception:
    logger.error("Unexpected error: ", exc_info=True)
