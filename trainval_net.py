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
from functools import partial

from cfgs.config import cfg
from data_handler.detection_data_manager import DetectionDataManager
from data_handler.data_manager_api import Mode
from loggers.tensorbord_logger import TensorBoardLogger
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_training_session import FasterRCNNTrainingSession

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

    data_manager_constructor = partial(DetectionDataManager, mode=Mode.TRAIN, imdb_name=cfg.imdb_name)
    logger = TensorBoardLogger(cfg.output_path)
    faster_rcnn = FasterRCNNTrainingSession(data_manager_constructor, FasterRCNNMetaArch, TensorBoardLogger, cfg)
    faster_rcnn.run_session()
