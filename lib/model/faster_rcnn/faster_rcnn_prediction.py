import os
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

def faster_rcnn_prediction(data_manager, model, cfg, epoch_num):
    num_images = len(data_manager)
    model.eval()
    raw_preds = {'bbox_coords': np.zeros((num_images, cfg.TEST.RPN_POST_NMS_TOP_N, model.num_predicted_coords)),
                 'cls_probs': np.zeros((num_images, cfg.TEST.RPN_POST_NMS_TOP_N, model.cfg_params['num_classes']))}

    pred_start = time.time()
    for i in range(20): #TODO: JA- loop until num_images
        im_data, im_info, gt_boxes, num_boxes = next(data_manager)
        curr_pred_start = time.time()
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
        faster_rcnn_loss_cls, faster_rcnn_loss_bbox, rois_label = \
            model(im_data, im_info, gt_boxes, num_boxes)

        cls_probs = cls_prob.data
        rpn_proposals = rois.data[:, :, 1:5]

        def transform_preds_to_img_coords():
            deltas_from_proposals = bbox_pred.data
            def unnormalize_preds():
                means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                unnormalized_deltas = deltas_from_proposals.view(-1, 4) * stds + means
                return unnormalized_deltas
            unnormalized_deltas = unnormalize_preds()
            reshaped_deltas = unnormalized_deltas.view(1, -1, model.num_predicted_coords)
            preds_in_img_coords = bbox_transform_inv(rpn_proposals, reshaped_deltas, 1)
            preds_clipped_to_img_size = clip_boxes(preds_in_img_coords, im_info.data, 1)
            inference_scaling_factor = im_info.data[0][2]
            bbox_coords = preds_clipped_to_img_size / inference_scaling_factor
            return bbox_coords
        bbox_coords = transform_preds_to_img_coords()
        cls_probs = cls_probs.squeeze()
        bbox_coords = bbox_coords.squeeze()
        curr_pred_end = time.time()
        pred_time = curr_pred_end - curr_pred_start
        avg_pred_time = (curr_pred_end - pred_start) / (i+1)
        logger.info('Prediction progress: {}/{}. Time for current image: {}. Avg time per image: {}.'.format(
            i+1, num_images, pred_time, avg_pred_time))
        raw_preds['bbox_coords'][i, ...] = bbox_coords.cpu().numpy()
        raw_preds['cls_probs'][i, ...] = cls_probs.cpu().numpy()

    preds_file_path = cfg.get_preds_path(epoch_num)
    os.makedirs(os.path.dirname(preds_file_path), exist_ok=True)
    with open(preds_file_path, 'wb') as f:
        pickle.dump(raw_preds, f, pickle.HIGHEST_PROTOCOL)

    pred_end = time.time()
    logger.info("Total prediction time: {}".format(pred_end - pred_start))
