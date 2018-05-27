import time
import logging
import numpy as np
import pickle

import torch

from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

from cfgs.config import cfg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: IB - the path to the saved model dir should be passed to the init. The init should load the model
# TODO: IB - currently this is in test_model, and look for the config file in the saved model dir.
# TODO: IB - if the config file doesn't exist - it should raise an exception and not use the global config file
# TODO: IB - another alternative is to save all the configs together with the model weights in the same file,
# TODO: IB - instead of in a separate config file - if possible

def faster_rcnn_postprocessing(data_manager, model, cfg, num_epoch):
    start_time = time.time()
    num_images = len(data_manager)
    num_classes = data_manager.num_classes
    preds_file_path = cfg.get_preds_path(epoch_num)
    postprocessed_detections = np.empty((num_images, num_classes), dtype='object')
    with open(preds_file_path, 'wb') as f:
        raw_preds = pickle.load(f)
    bbox_coords = raw_preds['bbox_coords']
    cls_probs = raw_preds['cls_probs']

    def keep_boxes_above_thresh_per_cls(probs, coords):
        nonzero_idxs = torch.nonzero(probs[:, j] > cfg.TEST.DETECTION_THRESH).view(-1)
        if nonzero_idxs.numel() > 0:
            filtered_probs = probs[:, j][nonzero_idxs]
            if model.cfg_params['is_class_agnostic']:
                filtered_coords = coords[nonzero_idxs, :]
            else:
                coord_idxs_cls_j = range(j * 4, (j + 1) * 4, 1)
                filtered_coords = coords[nonzero_idxs][:, coord_idxs_cls_j]
        else:
            filtered_coords = np.array([])
            filtered_probs = np.array([])
        return filtered_coords, filtered_probs

    def run_nms_on_thresholded_boxes(filtered_probs, filtered_coords):
        _, sorted_probs_idxs = torch.sort(filtered_probs, 0, True)
        detections_to_keep = torch.cat((filtered_coords, filtered_probs.unsqueeze(1)), 1)
        detections_to_keep = detections_to_keep[sorted_probs_idxs]
        idxs_to_keep = nms(detections_to_keep, cfg.TEST.NMS)
        detections_to_keep = detections_to_keep[idxs_to_keep.view(-1).long()]
        return detections_to_keep

    def keep_top_k_detections_in_image(image_detections):
        k = cfg.TEST.max_per_image
        if k > 0:
            cls_probs_per_image = np.hstack([image_detections[i, c][:, -1]
                                      for c in range(1, num_classes)])
            if len(cls_probs_per_image) > k:
                prob_thresh = np.sort(cls_probs_per_image)[-k]
                for c in range(1, num_classes):
                    boxes_idxs_to_keep = np.where(image_detections[i, c][:, -1] >= prob_thresh)[0]
                    image_detections[i, c] = image_detections[i, c][boxes_idxs_to_keep, :]
        return image_detections


    for i in range(20): #TODO: JA - change to num_images
        curr_coords = bbox_coords[i]
        curr_cls_probs = cls_probs[i]
        pp_start = time.time()
        for j in range(1, num_classes):
            coords_after_thresh, probs_after_thresh = keep_boxes_above_thresh_per_cls(
                curr_cls_probs, curr_coords)
            detections_after_nms = run_nms_on_thresholded_boxes(probs_after_thresh, coords_after_thresh)
            postprocessed_detections[i, j] = detections_after_nms.cpu().numpy()
        #todo: continue from here
        final_detections = keep_top_k_detections_in_image(postprocessed_detections)
        pp_end = time.time()
        logger.info('Postprocessing progress: {}/{}. Time for current image: {}. Avg time per image: {}.'.format(
            i, num_images, pp_end - pp_start, (pp_end-start_time) / i))

    pp_preds_path = cfg.get_postprocessed_preds_path(num_epoch)
    with open(pp_preds_path, 'wb') as f:
        pickle.dump(final_detections, f,)

    end_time = time.time()
    logger.info("Total prediction time - {}".format(end_time - start_time))

