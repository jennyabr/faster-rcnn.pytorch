# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from data_handler.detection_data_manager import FasterRCNNDataManager
from data_handler.data_manager_api import Mode
from loggers.tensorbord_logger import TensorBoardLogger
from model.faster_rcnn.ckpt_utils import load_session_from_ckpt
from model.faster_rcnn.faster_rcnn_training_session import run_training_session


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--resume_from', dest='ckpt_path', help='Path to ckpt file', type=str)
    args = parser.parse_args()

    if not args.resume_from:
        raise Exception("Unable to run without ckpt file.")

    model, optimizer_creation_fn, cfg, last_performed_epoch = load_session_from_ckpt(args.ckpt_path)
    global cfg

    train_data_manager = FasterRCNNDataManager(mode=Mode.TRAIN, imdb_name=cfg.imdb_name,
                                               seed=cfg.RNG_SEED, num_workers=cfg.NUM_WORKERS, is_cuda=cfg.CUDA,
                                               batch_size=cfg.TRAIN.batch_size)

    train_logger = TensorBoardLogger(cfg.output_path)

    run_training_session(train_data_manager, model, optimizer_creation_fn, cfg, train_logger,
                         first_epoch=last_performed_epoch + 1)
