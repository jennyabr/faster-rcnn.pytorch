import time
import os
import logging
import numpy as np
import torch
import torch.nn as nn

from model.feature_extractors.feature_extractors_factory import FeatureExtractorsFactory
from model.utils.net_utils import decay_lr_in_optimizer, clip_gradient
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.feature_extractors.resnet_for_faster_rcnn import ResNetForFasterRCNN
from model.feature_extractors.vgg16_for_faster_rcnn import VGG16ForFasterRCNN
from data_handler.data_with_test import DataPrep, Mode

from cfgs.config import cfg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FasterRCNN_Evaluator(object):
    def __init__(self, data_manager, trained_model_path):
        # TODO: IB - the path to the saved model dir should be passed to the init. The init should load the model
        # TODO: IB - currently this is in test_model, and look for the config file in the saved model dir.
        # TODO: IB - if the config file doesn't exist - it should raise an exception and not use the global config file
        # TODO: IB - another alternative is to save all the configs together with the model weights in the same file,
        # TODO: IB - instead of in a separate config file - if possible
        self.data_manager = data_manager
        feature_extractors = FeatureExtractorsFactory(cfg.net)
        self.faster_rcnn = FasterRCNNMetaArch(
                              feature_extractors,
                              class_names=self.data_obj.imdb.classes,
                              is_class_agnostic=cfg.class_agnostic,
                              num_regression_outputs_per_bbox=4,
                              roi_pooler_name=cfg.POOLING_MODE)

        logger.info("Loading checkpoint from {}.".format(trained_model_path))

        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']  # TODO

    def test_model(self):
        if cfg.CUDA:
            self.faster_rcnn.cuda()

        start = time.time()
        max_per_image = 100
        num_images = len(self.data_manager)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self.data_manager.num_classes)]

        output_dir = cfg.output_path  # TODO save to model dir?

        _t = {'im_detect': time.time(), 'misc': time.time()}
        det_file = os.path.join(output_dir, 'detections.pkl')

        self.faster_rcnn.eval()
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        for i in range(num_images):
            im_data, im_info, gt_boxes, num_boxes = next(self.data_manager)
            det_tic = time.time()
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, rois_label = self.faster_rcnn(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if cfg.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * self.data_manager.num_classes)

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2]



            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in range(1, self.data_manager.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, self.data_manager.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            logger.info('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'.format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                cv2.imwrite('result.png', im2show)
                pdb.set_trace()
                # cv2.imshow('test', im2show)
                # cv2.waitKey(0)

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        logger.info('Evaluating detections')
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        logger.info("test time: %0.4fs" % (end - start))