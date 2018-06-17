import logging
import pickle
import time

import numpy as np
import os
import torch

from model.meta_architecture.nms.nms_wrapper import nms

logger = logging.getLogger(__name__)


def faster_rcnn_postprocessing(data_manager, model, cfg, num_epoch):
    start_time = time.time()
    num_images = len(data_manager)
    num_classes = data_manager.num_classes
    preds_file_path = cfg.get_preds_path(num_epoch)
    logger.info("--->>> Starting postprocessing from :{}".format(preds_file_path))
    with open(preds_file_path, 'rb') as f:
        raw_preds = pickle.load(f)
    bbox_coords = raw_preds['bbox_coords']
    cls_probs = raw_preds['cls_probs']

    bbox_coords = torch.from_numpy(bbox_coords).cuda()
    cls_probs = torch.from_numpy(cls_probs).cuda()

    def keep_boxes_above_thresh_per_cls(probs, coords, curr_cls):
        nonzero_idxs = torch.nonzero(probs[:, curr_cls] > cfg.TEST.DETECTION_THRESH)
        nonzero_idxs = nonzero_idxs.view(-1)
        if nonzero_idxs.numel() > 0:
            filtered_probs = probs[:, curr_cls][nonzero_idxs]
            if model.cfg_params['is_class_agnostic']:
                filtered_coords = coords[nonzero_idxs, :]
            else:
                coord_idxs_curr_cls = range(curr_cls * 4, (curr_cls + 1) * 4, 1)
                filtered_coords = coords[nonzero_idxs][:, coord_idxs_curr_cls]
        else:
            filtered_coords = np.array([[]])
            filtered_probs = np.array([[]])
        return filtered_coords, filtered_probs

    def run_nms_on_unsorted_boxes(probs, coords):
        if probs.size == 0:
            return np.transpose(np.array([[], [], [], [], []]))
        _, sorted_probs_idxs = torch.sort(probs, 0, True)
        detections_to_keep = torch.cat((coords, probs.unsqueeze(1)), 1)
        detections_to_keep = detections_to_keep[sorted_probs_idxs]
        idxs_to_keep = nms(detections_to_keep, cfg.TEST.NMS)
        detections_to_keep = detections_to_keep[idxs_to_keep.view(-1).long()]
        return detections_to_keep.cpu().numpy()

    def keep_top_k_detections_in_image(image_detections):
        k = cfg.TEST.max_per_image
        if k > 0:
            cls_probs_per_image = np.hstack([image_detections[c][:, -1] for c in range(1, num_classes)])
            if len(cls_probs_per_image) > k:
                prob_thresh = np.sort(cls_probs_per_image)[-k]
                for c in range(1, num_classes):
                    boxes_idxs_to_keep = np.where(image_detections[c][:, -1] >= prob_thresh)[0]
                    image_detections[c] = image_detections[c][boxes_idxs_to_keep, :]
        return image_detections

    postprocessed_detections = np.empty(shape=(num_classes, num_images), dtype='object')
    for i in range(num_images):
        curr_coords = bbox_coords[i]
        curr_cls_probs = cls_probs[i]
        pp_start = time.time()
        detections_after_nms = np.empty(num_classes, dtype='object')
        for j in range(1, num_classes):
            coords_after_thresh, probs_after_thresh = keep_boxes_above_thresh_per_cls(curr_cls_probs, curr_coords, j)
            detections_after_nms[j] = run_nms_on_unsorted_boxes(probs_after_thresh, coords_after_thresh)
        postprocessed_detections[:, i] = keep_top_k_detections_in_image(detections_after_nms)
        pp_end = time.time()

        if i % cfg.TRAIN.disp_interval == 0 and i > 0:
            logger.info('Postprocessing in-progress: {0}/{1}: '
                        'avg time per image: {2:.4f} s.'.format(i, num_images, (pp_end-start_time) / (i+1)))

    pp_dets_path = cfg.get_postprocessed_detections_path(num_epoch)
    with open(pp_dets_path, 'wb') as f:
        pickle.dump(postprocessed_detections, f)

    end_time = time.time()
    logger.info("----------- Total postprocessing time {:.4f}s. -----------".format(end_time - start_time))

