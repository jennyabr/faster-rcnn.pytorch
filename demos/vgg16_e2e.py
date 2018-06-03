from __future__ import absolute_import
from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import logging
import numpy as np
import torch
from functools import partial

from data_manager.data_manager_abstract import Mode
from data_manager.classic_detection.classic_data_manager import ClassicDataManager
from loggers.tensorbord_logger import TensorBoardLogger
from pipeline.faster_rcnn.faster_rcnn_evaluation import faster_rcnn_evaluation
from model.faster_rcnn import FasterRCNN
from pipeline.faster_rcnn.faster_rcnn_training_session import run_training_session
from pipeline.faster_rcnn.faster_rcnn_postprocessing import faster_rcnn_postprocessing
from pipeline.faster_rcnn.faster_rcnn_prediction import faster_rcnn_prediction
from pipeline.faster_rcnn.faster_rcnn_visualization import faster_rcnn_visualization
from util.config import ConfigProvider
from util.logging import set_root_logger


config_file = '/home/jenny/gripper2/test_on_p100/cfgs/vgg16.yml'

cfg = ConfigProvider()
cfg.load(config_file)
np.random.seed(cfg.RNG_SEED)

set_root_logger(cfg.get_log_path())
logger = logging.getLogger(__name__)


def create_and_train():
    train_data_manager = ClassicDataManager(
        mode=Mode.TRAIN, imdb_name=cfg.imdb_name, num_workers=cfg.NUM_WORKERS, is_cuda=cfg.CUDA, cfg=cfg,
        batch_size=cfg.TRAIN.batch_size)

    train_logger = TensorBoardLogger(cfg.output_path)

    feature_extractors = create_feature_extractor_from_ckpt(
        cfg.net, cfg.net_variant, frozen_blocks=cfg.TRAIN.frozen_blocks,
        pretrained_model_path=cfg.TRAIN.get("pretrained_model_path", None))

    model = FasterRCNN.create_with_random_normal_init(feature_extractors, cfg,
                                                              num_classes=train_data_manager.num_classes)

    create_optimizer_fn = partial(torch.optim.SGD, momentum=cfg.TRAIN.MOMENTUM)

    run_training_session(train_data_manager, model, create_optimizer_fn, cfg, train_logger, cfg.TRAIN.start_epoch)


def pred_eval(epoch_nem):
    ckpt_path = cfg.get_last_ckpt_path()
    epoch_num = get_epoch_num_from_ckpt(ckpt_path)
    model = FasterRCNN.create_from_ckpt(ckpt_path)
    model.cuda()
    data_manager = ClassicDataManager(mode=Mode.INFER,
                                         imdb_name=cfg.imdbval_name,
                                         num_workers=cfg.NUM_WORKERS,
                                         is_cuda=cfg.CUDA,
                                         batch_size=cfg.TRAIN.batch_size,
                                         cfg=cfg)

    faster_rcnn_prediction(data_manager, model, cfg, epoch_nem)

    faster_rcnn_postprocessing(data_manager, model, cfg, epoch_nem)

    faster_rcnn_evaluation(data_manager, cfg, epoch_num)

    faster_rcnn_visualization(data_manager, cfg, epoch_nem)


try:
    create_and_train()
    pred_eval(epoch_nem=7)
except Exception:
    logger.error("Unexpected error: ", exc_info=True)
