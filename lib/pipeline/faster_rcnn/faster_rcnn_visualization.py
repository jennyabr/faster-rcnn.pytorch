import logging
import pickle
import time

import cv2
import numpy as np
import os

logger = logging.getLogger(__name__)


def faster_rcnn_visualization(data_manager, cfg, epoch_num):

    pp_preds_path = cfg.get_postprocessed_detections_path(epoch_num)
    logger.info("--->>> Starting visualization, reading post-processing data from: {}.".format(pp_preds_path))
    with open(pp_preds_path, 'rb') as f:
        bboxes = pickle.load(f)

    visualizations_dir = os.path.dirname(cfg.get_img_visualization_path(epoch_num, 0))
    start_time = time.time()
    for i in range(data_manager.num_images):
        im = cv2.imread(data_manager.imdb.image_path_at(i))
        im2show = np.copy(im)
        for j in range(1, data_manager.num_classes):
            cls_bboxes = bboxes[j, i]
            n_bboxes_to_visualize = np.minimum(10, cls_bboxes.shape[0])
            for bbox_ind in range(n_bboxes_to_visualize):
                bbox_coords = tuple(int(np.round(coords)) for coords in cls_bboxes[bbox_ind, :4])
                bbox_score = cls_bboxes[bbox_ind, -1]
                if bbox_score > 0.3:
                    cv2.rectangle(im2show, bbox_coords[0:2], bbox_coords[2:4], (0, 204, 0), 2)

                    class_name = data_manager.imdb.classes[j]
                    cv2.putText(im2show,
                                '{0}: {1:.3f}'.format(class_name, bbox_score),
                                (bbox_coords[0] + 15, bbox_coords[1]),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.0,
                                (0, 0, 255),
                                thickness=1)

        cv2.imwrite(cfg.get_img_visualization_path(epoch_num, i), im2show)
        if i % cfg.TEST.disp_interval == 0 and i > 0:
            logger.info("Visualization in-progress: {}/{}.".format(i, data_manager.num_images))

    end_time = time.time()
    logger.info("Visualization dir path: {}.".format(visualizations_dir))
    logger.info("-------------- Visualization time: {:.4f} s. --------------".format(end_time - start_time))
