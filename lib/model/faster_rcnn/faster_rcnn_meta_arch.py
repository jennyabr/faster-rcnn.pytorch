import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable

from model.feature_extractors.faster_rcnn_feature_extractor_duo import create_empty_duo
from model.roi_poolers.roi_pooler_factory import create_roi_pooler
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.net_utils import _smooth_l1_loss, normal_init
from util.config import ConfigProvider


class FasterRCNNMetaArch(nn.Module):
    def __init__(self, feature_extractor_duo, cfg, num_classes):
        super(FasterRCNNMetaArch, self).__init__()
        cfg_params = {'num_classes': num_classes,
                      'is_class_agnostic': cfg.class_agnostic,
                      'roi_pooler_name': cfg.roi_pooler_name,
                      'roi_pooler_size': cfg.roi_pooler_size,
                      'crop_resize_with_max_pool': cfg.CROP_RESIZE_WITH_MAX_POOL,
                      'num_regression_outputs_per_bbox': cfg.num_regression_outputs_per_bbox}

        self.cfg_params = cfg_params

        def create_rpn():
            rpn_fe = feature_extractor_duo.rpn_feature_extractor
            rpn_fe_output_depth = rpn_fe.output_num_channels
            rpn_and_nms = _RPN(rpn_fe_output_depth, cfg)
            # TODO JA - the ProposalTargetLayer is not intuitive
            rpn_proposal_target = _ProposalTargetLayer(cfg_params['num_classes'], cfg)
            return rpn_fe, rpn_and_nms, rpn_proposal_target

        self.rpn_fe, self.rpn_and_nms, self.rpn_proposal_target = create_rpn()

        self.roi_pooler = create_roi_pooler(cfg_params['roi_pooler_name'], cfg_params['roi_pooler_size'],
                                            cfg_params['crop_resize_with_max_pool'])

        def create_fast_rcnn():
            fast_rcnn_fe = feature_extractor_duo.fast_rcnn_feature_extractor
            fast_rcnn_fe_output_depth = fast_rcnn_fe.output_num_channels
            if cfg_params['is_class_agnostic']:
                self.num_predicted_coords = cfg_params['num_regression_outputs_per_bbox']
            else:
                self.num_predicted_coords = cfg_params['num_regression_outputs_per_bbox'] * \
                                            cfg_params['num_classes']
            bbox_head = nn.Linear(fast_rcnn_fe_output_depth, self.num_predicted_coords)
            fast_rcnn_bbox_head = bbox_head

            fast_rcnn_cls_head = nn.Linear(fast_rcnn_fe_output_depth, cfg_params['num_classes'])

            return fast_rcnn_fe, fast_rcnn_bbox_head, fast_rcnn_cls_head
        self.fast_rcnn_feature_extractor, self.fast_rcnn_bbox_head, self.fast_rcnn_cls_head = create_fast_rcnn()

        # TODO JA - should the loss be in self?
        self.faster_rcnn_loss_cls = 0
        self.faster_rcnn_loss_bbox = 0

    @classmethod
    def create_with_random_normal_init(cls, feature_extractor_duo, cfg, num_classes):
        faster_rcnn = cls(feature_extractor_duo, cfg, num_classes)
        configured_normal_init = partial(normal_init, mean=0)
        configured_normal_init(faster_rcnn.rpn_and_nms.RPN_Conv, stddev=0.01)
        configured_normal_init(faster_rcnn.rpn_and_nms.RPN_cls_score, stddev=0.01)
        configured_normal_init(faster_rcnn.rpn_and_nms.RPN_bbox_pred, stddev=0.01)
        configured_normal_init(faster_rcnn.fast_rcnn_cls_head, stddev=0.01)
        configured_normal_init(faster_rcnn.fast_rcnn_bbox_head, stddev=0.001)
        return faster_rcnn

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        base_feature_map = self.rpn_fe(im_data)

        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn_and_nms(base_feature_map, im_info, gt_boxes, num_boxes)

        # TODO: JA - this if-else was skipped because we didn't want to dive into rpn_proposal_target
        if self.training:  # if it is training phrase, then use ground truth bboxes for refining
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = \
                self.rpn_proposal_target(rois, gt_boxes, num_boxes)

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        pooled_rois = self.roi_pooler(base_feature_map, rois.view(-1, 5))

        def run_fast_rcnn():
            fast_rcnn_feature_map = self.fast_rcnn_feature_extractor(pooled_rois)
            bbox_pred = self.fast_rcnn_bbox_head(fast_rcnn_feature_map)
            if self.training and not self.cfg_params['is_class_agnostic']:
                # select the corresponding columns according to roi labels
                # TODO JA: these 4s int the next line might be hardcoding the number of output coords
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view,
                                                1,
                                                rois_label.view(rois_label.size(0), 1, 1)
                                                .expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

            cls_score = self.fast_rcnn_cls_head(fast_rcnn_feature_map)
            cls_prob = F.softmax(cls_score, dim=-1)
            return bbox_pred, cls_score, cls_prob
        bbox_pred, cls_score, cls_prob = run_fast_rcnn()

        if self.training:
            self.faster_rcnn_loss_cls = F.cross_entropy(cls_score, rois_label)
            self.faster_rcnn_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        else:
            self.faster_rcnn_loss_cls = 0
            self.faster_rcnn_loss_bbox = 0

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
               self.faster_rcnn_loss_cls, self.faster_rcnn_loss_bbox, rois_label

    @classmethod
    def create_from_ckpt(cls, ckpt_path):
        # TODO: JA - enable manually overriding num_classes and enable to randomize the last layers
        state_dict = torch.load(os.path.abspath(ckpt_path))
        loaded_cfg = ConfigProvider()
        loaded_cfg.create_from_dict(state_dict['ckpt_cfg'])
        feature_extractor_duo = create_empty_duo(
            loaded_cfg.net, loaded_cfg.net_variant, loaded_cfg.TRAIN.frozen_blocks)
        model = FasterRCNNMetaArch(feature_extractor_duo, loaded_cfg, state_dict['model_cfg_params']['num_classes'])
        model.load_state_dict(state_dict['model'])
        return model

    def train(self, mode=True):
        super(FasterRCNNMetaArch, self).train(mode)