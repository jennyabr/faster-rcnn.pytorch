import pickle
import time

from cfgs.config import get_logger


def faster_rcnn_evaluation(data_manager, cfg, detections_path, eval_dir_path):  # TODO JA alternative prams
    logger = get_logger(__name__)
    logger.info(' --->>> Evaluating detections from: {}'.format(detections_path))
    with open(detections_path, 'rb') as f:
        dets_to_evaluate = pickle.load(f)

    start_time = time.time()
    data_manager.imdb.evaluate_detections(dets_to_evaluate, eval_dir_path)
    end_time = time.time()
    logger.info(" ---------- Evaluating detections time: {:.4f}s. ---------- ".format(end_time - start_time))

