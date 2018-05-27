import logging
from functools import partial
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.feature_extractors.faster_rcnn_feature_extractors import create_feature_extractor_empty
from model.roi_poolers.roi_pooler_factory import create_roi_pooler
from model.rpn.rpn import _RPN
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _affine_grid_gen, normal_init

from cfgs.config import cfg, ConfigProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FasterRCNNMetaArch(nn.Module):
    def __init__(self, feature_extractors, cfg, num_classes):
        super(FasterRCNNMetaArch, self).__init__()
        cfg_params = {'num_classes': num_classes,
                      'is_class_agnostic': cfg.class_agnostic,
                      'roi_pooler_name': cfg.roi_pooler_name,
                      'roi_pooler_size': cfg.roi_pooler_size,
                      'crop_resize_with_max_pool': cfg.CROP_RESIZE_WITH_MAX_POOL,
                      'num_regression_outputs_per_bbox': cfg.num_regression_outputs_per_bbox}

        self.cfg_params = cfg_params
        self.base_feature_extractor = feature_extractors.base_feature_extractor

        def create_rpn():
            rpn_fe_output_depth = feature_extractors.get_output_num_channels(self.base_feature_extractor)
            rpn_and_nms = _RPN(rpn_fe_output_depth)
            # TODO: the ProposalTargetLayer is not intuitive
            rpn_proposal_target = _ProposalTargetLayer(cfg_params['num_classes'])
            return rpn_and_nms, rpn_proposal_target

        self.rpn_and_nms, self.rpn_proposal_target = create_rpn()

        self.roi_pooler = create_roi_pooler(cfg_params['roi_pooler_name'])
        # TODO delete next line:
        self.grid_size = cfg_params['roi_pooler_size'] * 2 if cfg_params['crop_resize_with_max_pool'] else\
            cfg_params['roi_pooler_size']

        def create_fast_rcnn():
            fast_rcnn_fe = feature_extractors.fast_rcnn_feature_extractor
            fast_rcnn_fe_output_depth = feature_extractors.get_output_num_channels(fast_rcnn_fe.feature_extractor)  #TODO this functioncan get any model...
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

        # TODO should the loss be in self?
        self.faster_rcnn_loss_cls = 0
        self.faster_rcnn_loss_bbox = 0


    @classmethod
    def create_with_random_normal_init(cls, feature_extractors, cfg, num_classes):
        faster_rcnn = cls(feature_extractors, cfg, num_classes)
        configured_normal_init = partial(normal_init, mean=0)
        configured_normal_init(faster_rcnn.rpn_and_nms.RPN_Conv, stddev=0.01)
        configured_normal_init(faster_rcnn.rpn_and_nms.RPN_cls_score, stddev=0.01)
        configured_normal_init(faster_rcnn.rpn_and_nms.RPN_bbox_pred, stddev=0.01)
        configured_normal_init(faster_rcnn.fast_rcnn_cls_head, stddev=0.01)
        configured_normal_init(faster_rcnn.fast_rcnn_bbox_head, stddev=0.001)
        return faster_rcnn


    #TODO JA - fix - add parameter to init random some layers and number of classes
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        base_feature_map = self.base_feature_extractor(im_data)

        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn_and_nms(base_feature_map, im_info, gt_boxes, num_boxes)

        # TODO: IB- we skipped this if-else because we didn't want to dive into rpn_proposal_target
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

        rois = Variable(rois) # TODO make immutable

        #TODO refactor this:
        if cfg.roi_pooler_name == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feature_map.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_rois = self.roi_pooler(base_feature_map, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_rois = F.max_pool2d(pooled_rois, 2, 2)
        elif cfg.roi_pooler_name == 'align':
            pooled_rois = self.roi_pooler(base_feature_map, rois.view(-1, 5))
        elif cfg.roi_pooler_name == 'pool':
            pooled_rois = self.roi_pooler(base_feature_map, rois.view(-1, 5))
        #TODO this insted : pooled_rois = self.roi_pooler(base_feature_map, rois.view(-1, 5))
        #TODO: IB - what is this 5 in the view? it might be hardcoding the number of bb coords

        def run_fast_rcnn():
            fast_rcnn_feature_map = self.fast_rcnn_feature_extractor(pooled_rois)
            bbox_pred = self.fast_rcnn_bbox_head(fast_rcnn_feature_map)
            if self.training and not self.cfg_params['is_class_agnostic']:
                # select the corresponding columns according to roi labels
                #  TODO: replace the comment with an encapsulating function
                # TODO: these 4s might be hardcoding the number of output coords
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view,
                                                1,
                                                rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

            cls_score = self.fast_rcnn_cls_head(fast_rcnn_feature_map)
            cls_prob = F.softmax(cls_score)
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

        return rois, cls_prob, bbox_pred, \
               rpn_loss_cls, rpn_loss_bbox, \
               self.faster_rcnn_loss_cls, self.faster_rcnn_loss_bbox, rois_label

    # TODO: JA - enable manually overriding num_classes and enable to randomize the last layers
    @classmethod
    def create_from_ckpt(cls, ckpt_path):
        state_dict = torch.load(os.path.expanduser(ckpt_path))
        loaded_cfg = ConfigProvider()
        loaded_cfg.create_from_dict(state_dict['ckpt_cfg'])
        feature_extractors = create_feature_extractor_empty(
            loaded_cfg.net, loaded_cfg.net_variant, loaded_cfg.TRAIN.frozen_blocks)
        model = FasterRCNNMetaArch(feature_extractors, loaded_cfg, state_dict['model_cfg_params']['num_classes'])
        model.load_state_dict(state_dict['model'])
        return model
