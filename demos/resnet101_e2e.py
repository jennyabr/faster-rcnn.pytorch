from __future__ import absolute_import
from __future__ import division
import os

from pipeline.faster_rcnn.run_functions.run_classic_pipeline import create_and_train_with_err_handling, \
    pred_eval_with_err_handling

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
import torch
from functools import partial

from data_manager.data_manager_abstract import Mode
from data_manager.classic_detection.classic_data_manager import ClassicDataManager
from loggers.tensorbord_logger import TensorBoardLogger
from pipeline.faster_rcnn.faster_rcnn_evaluation import faster_rcnn_evaluation
from model.meta_architecture.faster_rcnn import FasterRCNN
from pipeline.faster_rcnn.faster_rcnn_training_session import run_training_session
from pipeline.faster_rcnn.faster_rcnn_postprocessing import faster_rcnn_postprocessing
from pipeline.faster_rcnn.faster_rcnn_prediction import faster_rcnn_prediction
from pipeline.faster_rcnn.faster_rcnn_visualization import faster_rcnn_visualization
from model.feature_extractors.feature_extractor_duo import create_duo_from_ckpt
from model.utils.misc_utils import get_epoch_num_from_ckpt

from util.config import ConfigProvider
from util.logging import set_root_logger


config_file = os.path.join(os.getcwd(), 'demos', 'cfgs', 'resnet101.yml')

cfg = ConfigProvider()
cfg.load(config_file)
create_and_train_with_err_handling(cfg)
pred_eval_with_err_handling(cfg)

