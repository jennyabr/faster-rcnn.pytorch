import time
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def faster_rcnn_evaluation(data_manager, cfg, epoch_num):  # TODO JA alternative prams
    logger.info('Evaluating detections.')
    pp_preds_path = cfg.get_postprocessed_preds_path(epoch_num)
    with open(pp_preds_path, 'rb') as f:
        processed_preds = pickle.load(f)

    start_time = time.time()
    data_manager.imdb.evaluate_detections(processed_preds, cfg.get_evals_dir_path(epoch_num))
    end_time = time.time()
    logger.info("Evaluating detections time: {:.4f}s".format(end_time - start_time))
