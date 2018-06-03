import argparse
import logging
import numpy as np

from data_handler.data_manager_api import Mode
from data_handler.detection_data_manager import FasterRCNNDataManager
from model.faster_rcnn.faster_rcnn_evaluation import faster_rcnn_evaluation
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_postprocessing import faster_rcnn_postprocessing
from model.faster_rcnn.faster_rcnn_prediction import faster_rcnn_prediction
from model.faster_rcnn.faster_rcnn_visualization import faster_rcnn_visualization
from model.utils.misc_utils import get_epoch_num_from_ckpt
from util.config import ConfigProvider
from util.logging import set_root_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--config_dir', dest='config_dir', help='Path to config dir', type=str)
    args = parser.parse_args()

    cfg = ConfigProvider()
    cfg.load(args.config_dir)
    np.random.seed(cfg.RNG_SEED)

    set_root_logger(cfg.get_log_path())
    logger = logging.getLogger(__name__)

    try:
        ckpt_path = cfg.get_last_ckpt_path()
        epoch_num = get_epoch_num_from_ckpt(ckpt_path)
        model = FasterRCNNMetaArch.create_from_ckpt(ckpt_path)
        model.cuda()
        data_manager = FasterRCNNDataManager(mode=Mode.INFER,
                                             imdb_name=cfg.imdbval_name,
                                             num_workers=cfg.NUM_WORKERS,
                                             is_cuda=cfg.CUDA,
                                             batch_size=cfg.TRAIN.batch_size,
                                             cfg=cfg)

        faster_rcnn_prediction(data_manager, model, cfg, epoch_num)

        faster_rcnn_postprocessing(data_manager, model, cfg, epoch_num)

        faster_rcnn_evaluation(data_manager, cfg, epoch_num)

        faster_rcnn_visualization(data_manager, cfg, epoch_num)

    except Exception:
        logger.error("Unexpected error: ", exc_info=True)
