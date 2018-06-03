import logging
import pickle
import time

logger = logging.getLogger(__name__)


def faster_rcnn_evaluation(data_manager, cfg, epoch_num):
    detections_path = cfg.get_postprocessed_detections_path(epoch_num)
    eval_dir_path = cfg.get_evals_dir_path(epoch_num)
    logger.info('--->>> Evaluating detections from: {}'.format(detections_path))
    with open(detections_path, 'rb') as f:
        dets_to_evaluate = pickle.load(f)

    start_time = time.time()
    data_manager.imdb.evaluate_detections(dets_to_evaluate, eval_dir_path)
    end_time = time.time()
    logger.info("---------- Evaluating detections time: {:.4f}s. ----------".format(end_time - start_time))

