import logging
import pickle
import time

import numpy as np
import os
import torch

from model.meta_architecture.rpn.bbox_transform import bbox_transform_inv, clip_boxes

logger = logging.getLogger(__name__)


def faster_rcnn_prediction(data_manager, model, cfg, epoch_num):
    logger.info("--->>> Starting prediction...")
    num_images = len(data_manager)
    model.eval()
    raw_preds = {
        'bbox_coords': np.zeros((num_images, cfg.TEST.RPN_POST_NMS_TOP_N, model.num_predicted_coords),
                                dtype=np.float32),
        'cls_probs': np.zeros((num_images, cfg.TEST.RPN_POST_NMS_TOP_N, model.cfg_params['num_classes']),
                              dtype=np.float32)
    }

    pred_start = time.time()
    data_manager.prepare_iter_for_new_epoch()
    for i in range(num_images):
        im_data, im_info, gt_boxes, num_boxes = next(data_manager)
        curr_pred_start = time.time()
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
            faster_rcnn_loss_cls, faster_rcnn_loss_bbox, rois_label = \
            model(im_data, im_info, gt_boxes, num_boxes)

        cls_probs = cls_prob.data
        rpn_proposals = rois.data[:, :, 1:5]

        def transform_preds_to_img_coords():

            def unnormalize_preds():
                deltas_from_proposals = bbox_pred.data  # TODO why data?
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

        raw_preds['bbox_coords'][i, ...] = bbox_coords.cpu().numpy()
        raw_preds['cls_probs'][i, ...] = cls_probs.cpu().numpy()

        if i % cfg.TRAIN.disp_interval == 0 and i > 0:
            logger.info('Prediction in-progress {0}/{1}: '
                        'avg per image: {2:.3f} s.'.format(i, num_images, avg_pred_time))

    preds_file_path = cfg.get_preds_path(epoch_num)
    with open(preds_file_path, 'wb') as f:
        pickle.dump(raw_preds, f, pickle.HIGHEST_PROTOCOL)

    pred_end = time.time()
    logger.info("------------ Total prediction time: {:.4f}s. -------------".format(pred_end - pred_start))
