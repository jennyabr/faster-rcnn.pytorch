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

def faster_rcnn_postprocessing(data_manager, model, cfg):
    start_time = time.time()
    num_images = len(data_manager)
    num_classes = data_manager.num_classes

    output_dir = cfg.get_eval_outputs_path()
    preds_file = cfg.get_preds_path()
    for i in range(num_images):
        im_data, im_info, gt_boxes, num_boxes = next(data_manager)
        pred_start = time.time()
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox = \
            model(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        rpn_proposals = rois.data[:, :, 1:5]

        def transform_preds_to_img_coords():
            deltas_from_proposals = bbox_pred.data
            def unnormalize_preds():
                means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                unnormalized_deltas = deltas_from_proposals.view(-1, 4) * stds + means
                return unnormalized_deltas
            unnormalized_deltas = unnormalize_preds(deltas_from_proposals)
            reshaped_deltas = unnormalized_deltas.view(1, -1, model.num_predicted_coords)
            preds_in_img_coords = bbox_transform_inv(rpn_proposals, reshaped_deltas, 1)
            preds_clipped_to_img_size = clip_boxes(preds_in_img_coords, im_info.data, 1)
            inference_scaling_factor = im_info[0][2]
            bbox_coords = preds_clipped_to_img_size / inference_scaling_factor
            return bbox_coords
        bbox_coords = transform_preds_to_img_coords()
        scores = scores.squeeze()
        bbox_coords = bbox_coords.squeeze()
        pred_end = time.time()
        pred_time = pred_end - pred_start
        postprocessing_start = time.time()
        for j in range(1, num_classes):
            inds = torch.nonzero(scores[:, j] > cfg.TEST.DETECTION_THRESH).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if cfg.TRAIN.class_agnostic:
                    cls_boxes = bbox_coords[inds, :]
                else:
                    cls_boxes = bbox_coords[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_prediction

        max_per_image = cfg.TEST.max_per_image
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[c][i][:, -1]
                                      for c in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for c in range(1, num_classes):
                    keep = np.where(all_boxes[c][i][:, -1] >= image_thresh)[0]
                    all_boxes[c][i] = all_boxes[c][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        logger.info('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))

    with open(cfg.get_predictions_path(), 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    end_time = time.time()
    logger.info("Prediction time: %0.4fs" % (end_time - start_time))
