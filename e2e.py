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
from model.faster_rcnn.faster_rcnn_evaluation import faster_rcnn_evaluation
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_postprocessing import faster_rcnn_postprocessing
from model.faster_rcnn.faster_rcnn_prediction import faster_rcnn_prediction
from model.faster_rcnn.faster_rcnn_training_session import run_training_session
from model.faster_rcnn.faster_rcnn_visualization import faster_rcnn_visualization
from model.feature_extractors.faster_rcnn_feature_extractors import create_feature_extractor_from_ckpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    predict_on_epoch = 6
    model = FasterRCNNMetaArch.create_from_ckpt(cfg.get_ckpt_path(predict_on_epoch))
    model.cuda()
    data_manager = FasterRCNNDataManager(mode=Mode.INFER,
                                              imdb_name=cfg.imdbval_name,
                                              seed=cfg.RNG_SEED,
                                              num_workers=cfg.NUM_WORKERS,
                                              is_cuda=cfg.CUDA,
                                              batch_size=cfg.TRAIN.batch_size)

    faster_rcnn_prediction(data_manager, model, cfg, predict_on_epoch)
    faster_rcnn_postprocessing(data_manager, model, cfg, predict_on_epoch)
    detections_path = cfg.get_postprocessed_detections_path(predict_on_epoch)
    eval_path = cfg.get_evals_dir_path(predict_on_epoch)
    faster_rcnn_evaluation(data_manager, cfg, detections_path, eval_path)
    faster_rcnn_visualization(data_manager, cfg, predict_on_epoch)

