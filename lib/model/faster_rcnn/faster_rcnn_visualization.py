import logging
import pickle
import time

import cv2
import numpy as np
import os


def faster_rcnn_visualization(data_manager, cfg, epoch_num):
    logger = logging.getLogger(__name__)

    pp_preds_path = cfg.get_postprocessed_detections_path(epoch_num)
    logger.info(" --->>> Reading post-processing data from: {}".format(pp_preds_path))
    with open(pp_preds_path, 'rb') as f:
        bboxes = pickle.load(f)

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
                                '%s: %.3f' % (class_name, bbox_score),
                                (bbox_coords[1], bbox_coords[0] + 15),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.0,
                                (0, 0, 255),
                                thickness=1)

        visualizations_path_img_i = cfg.get_img_visualization_path(epoch_num, i)
        os.makedirs(os.path.dirname(visualizations_path_img_i), exist_ok=True)
        logger.info("Writing image {} visualization to: {}.".format(i, visualizations_path_img_i))
        cv2.imwrite(visualizations_path_img_i, im2show)

    end_time = time.time()
    logger.info(" -------------- Visualization time: {:.4f}s.  -------------- ".format(end_time - start_time))
