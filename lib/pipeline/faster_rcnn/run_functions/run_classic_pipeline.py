import logging

import torch
from functools import partial

from data_manager.classic_detection.classic_data_manager import ClassicDataManager
from data_manager.data_manager_abstract import Mode
from loggers.tensorbord_logger import TensorBoardLogger
from model.feature_extractors.feature_extractor_duo import create_duo_from_ckpt
from model.meta_architecture.faster_rcnn import FasterRCNN
from model.utils.misc_utils import get_epoch_num_from_ckpt
from pipeline.faster_rcnn.faster_rcnn_evaluation import faster_rcnn_evaluation
from pipeline.faster_rcnn.faster_rcnn_postprocessing import faster_rcnn_postprocessing
from pipeline.faster_rcnn.faster_rcnn_prediction import faster_rcnn_prediction
from pipeline.faster_rcnn.faster_rcnn_training_session import run_training_session
from pipeline.faster_rcnn.faster_rcnn_visualization import faster_rcnn_visualization


logger = logging.getLogger(__name__)


def create_and_train_with_err_handling(cfg):
    try:
        create_and_train(cfg)
    except Exception as e:
        logger.error("Unexpected error: ", exc_info=True)
        raise e


def create_and_train(cfg):
    train_data_manager = ClassicDataManager(
        mode=Mode.TRAIN, imdb_name=cfg.imdb_name, num_workers=cfg.NUM_WORKERS, is_cuda=cfg.CUDA, cfg=cfg,
        batch_size=cfg.TRAIN.batch_size)

    train_logger = TensorBoardLogger(cfg.output_path)

    feature_extractor_duo = create_duo_from_ckpt(
        cfg.net, cfg.net_variant, frozen_blocks=cfg.TRAIN.frozen_blocks,
        pretrained_model_path=cfg.TRAIN.get("pretrained_model_path", None))

    model = FasterRCNN.create_with_random_normal_init(feature_extractor_duo, cfg,
                                                      num_classes=train_data_manager.num_classes)

    create_optimizer_fn = partial(torch.optim.SGD, momentum=cfg.TRAIN.MOMENTUM)

    run_training_session(train_data_manager, model, create_optimizer_fn, cfg, train_logger, cfg.TRAIN.start_epoch)


def pred_eval_with_err_handling(cfg):
    try:
        pred_eval(cfg)
    except Exception as e:
        logger.error("Unexpected error: ", exc_info=True)
        raise e


def pred_eval(cfg):
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

    faster_rcnn_prediction(data_manager, model, cfg, epoch_num)

    faster_rcnn_postprocessing(data_manager, model, cfg, epoch_num)

    faster_rcnn_evaluation(data_manager, cfg, epoch_num)

    faster_rcnn_visualization(data_manager, cfg, epoch_num)
