import time
import logging
import numpy as np
import pickle

import torch
from torch.autograd import Variable
from model.nms.nms_wrapper import nms


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def faster_rcnn_postprocessing(data_manager, model, cfg, num_epoch):
    logger.info("Starting postprocessing.")
    start_time = time.time()
    num_images = 20
    # num_images = len(data_manager) #TODO: JA - uncomment this
    num_classes = data_manager.num_classes
    preds_file_path = cfg.get_preds_path(num_epoch)
    postprocessed_detections = np.empty(num_images, dtype='object')
    with open(preds_file_path, 'rb') as f:
        raw_preds = pickle.load(f)
    bbox_coords = raw_preds['bbox_coords']
    cls_probs = raw_preds['cls_probs']

    bbox_coords = torch.from_numpy(bbox_coords)
    cls_probs = torch.from_numpy(cls_probs)


    def keep_boxes_above_thresh_per_cls(probs, coords):
        nonzero_idxs = torch.nonzero(probs[:, j] > cfg.TEST.DETECTION_THRESH)
        nonzero_idxs = nonzero_idxs.view(-1)
        if nonzero_idxs.numel() > 0:
            filtered_probs = probs[:, j][nonzero_idxs]
            if model.cfg_params['is_class_agnostic']:
                filtered_coords = coords[nonzero_idxs, :]
            else:
                coord_idxs_cls_j = range(j * 4, (j + 1) * 4, 1)
                filtered_coords = coords[nonzero_idxs][:, coord_idxs_cls_j]
        else:
            filtered_coords = np.array([[] * 4])  # TODO
            filtered_probs = np.array([[] * num_classes])  # TODO
        return filtered_coords, filtered_probs

    def run_nms_on_unsorted_boxes(probs, coords):
        _, sorted_probs_idxs = torch.sort(probs, 0, True)
        detections_to_keep = torch.cat((coords, probs.unsqueeze(1)), 1)
        detections_to_keep = detections_to_keep[sorted_probs_idxs]
        idxs_to_keep = nms(detections_to_keep, cfg.TEST.NMS)
        detections_to_keep = detections_to_keep[idxs_to_keep.view(-1).long()]
        return detections_to_keep

    def keep_top_k_detections_in_image(image_detections):
        k = cfg.TEST.max_per_image
        if k > 0:
            cls_probs_per_image = np.hstack([image_detections[i, c][:, -1] for c in range(1, num_classes)])
            if len(cls_probs_per_image) > k:
                prob_thresh = np.sort(cls_probs_per_image)[-k]
                for c in range(1, num_classes):
                    boxes_idxs_to_keep = np.where(image_detections[i, c][:, -1] >= prob_thresh)[0]
                    image_detections[i, c] = image_detections[i, c][boxes_idxs_to_keep, :]
        return image_detections

    for i in range(20):  # TODO: JA - change to num_images
        curr_coords = bbox_coords[i]
        curr_cls_probs = cls_probs[i]
        pp_start = time.time()
        detections_after_nms = np.empty(num_classes, dtype='object')
        for j in range(1, num_classes):
            coords_after_thresh, probs_after_thresh = keep_boxes_above_thresh_per_cls(curr_cls_probs, curr_coords)
            detections_after_nms[j] = run_nms_on_unsorted_boxes(probs_after_thresh, coords_after_thresh).cpu().numpy()
        postprocessed_detections[i] = keep_top_k_detections_in_image(detections_after_nms)
        pp_end = time.time()
        logger.info('Postprocessing progress: {0}/{1}: '
                    'Time for current image: {2:.4f}s '
                    '[Avg time per image: {3:.4f}s].'.format(i, num_images, pp_end - pp_start, (pp_end-start_time) / i))

    pp_preds_path = cfg.get_postprocessed_preds_path(num_epoch)
    with open(pp_preds_path, 'wb') as f:
        pickle.dump(postprocessed_detections, f)

    end_time = time.time()
    logger.info("Total postprocessing time {:.4f}s.".format(end_time - start_time))

