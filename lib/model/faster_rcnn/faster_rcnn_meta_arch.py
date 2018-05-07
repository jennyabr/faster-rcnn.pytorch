import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.feature_extractors.vgg16_for_faster_rcnn import FasterRCNNFeatureExtractors
from lib.model.roi_poolers.roi_pooler_factory import create_roi_pooler
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss, _affine_grid_gen


class FasterRCNNMetaArch(nn.Module):
    def __init__(self, faster_rcnn_feature_extractors, class_names,
                 predict_bbox_per_class=False, num_regression_outputs_per_bbox=4, roi_pooler_name='crop'):
        super(FasterRCNNMetaArch, self).__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.predict_bbox_per_class = predict_bbox_per_class

        self.base_feature_extractor = faster_rcnn_feature_extractors.get_base_feature_extractor()
        def create_rpn():
            rpn_FE_output_depth = FasterRCNNFeatureExtractors.get_output_num_channels(self.base_feature_extractor)
            rpn_and_nms = _RPN(rpn_FE_output_depth)
            rpn_proposal_target = _ProposalTargetLayer(self.num_classes) #TODO: the ProposalTargetLayer is not intuitive
            return rpn_and_nms, rpn_proposal_target
        self.rpn_and_nms, self.rpn_proposal_target = create_rpn()

        self.roi_pooler = create_roi_pooler(roi_pooler_name)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE #TODO delete this linbe

        def create_fast_rcnn():
            fast_rcnn_feature_extractor = faster_rcnn_feature_extractors.get_fast_rcnn_feature_extractor()
            fast_rcnn_FE_output_depth = FasterRCNNFeatureExtractors.get_output_num_channels(fast_rcnn_feature_extractor.feature_extractor)
            assert fast_rcnn_FE_output_depth == 4096 #TODO delete this assertion

            if self.predict_bbox_per_class:
                bbox_head = nn.Linear(fast_rcnn_FE_output_depth, num_regression_outputs_per_bbox * self.num_classes)
            else:
                bbox_head = nn.Linear(fast_rcnn_FE_output_depth, num_regression_outputs_per_bbox)
            fast_rcnn_heads = {'bbox': bbox_head, 'cls': nn.Linear(fast_rcnn_FE_output_depth, self.num_classes)}
            return fast_rcnn_feature_extractor, fast_rcnn_heads
        self.fast_rcnn_feature_extractor, self.fast_rcnn_heads = create_fast_rcnn()

        faster_rcnn_loss_cls = 0
        faster_rcnn_loss_bbox = 0

        def init_weights(rpn, bbox_head, cls_head):
            def normal_init(m, mean, stddev, truncated=False):
                if truncated:
                    m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
                else:
                    m.weight.data.normal_(mean, stddev)
                    m.bias.data.zero_()
            normal_init(rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(cls_head, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(bbox_head, 0, 0.001, cfg.TRAIN.TRUNCATED)
        init_weights(self.rpn_and_nms, self.fast_rcnn_heads['bbox'], self.fast_rcnn_heads['cls'])

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

        rois = Variable(rois)

        #TODO delete this:
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feature_map.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.roi_pooler(base_feature_map, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_rois = self.roi_pooler(base_feature_map, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_rois = self.roi_pooler(base_feature_map, rois.view(-1, 5))

        #TODO this insted : pooled_rois = self.roi_pooler(base_feature_map, rois.view(-1, 5))



        #TODO: IB - what is this 5 in the view? it might be hardcoding the number of bb coords
        def run_fast_rcnn():
            fast_rcnn_feature_map = self.fast_rcnn_feature_extractor(pooled_rois)
            bbox_pred = self.fast_rcnn_heads['bbox'](fast_rcnn_feature_map)
            if self.training and self.predict_bbox_per_class:
                # select the corresponding columns according to roi labels # TODO: replace the comment with an encapsulating function
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4) #TODO: these 4s might be hardcoding the number of output coords
                bbox_pred_select = torch.gather(bbox_pred_view,
                                                1,
                                                rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

            cls_score = self.fast_rcnn_heads['cls'](fast_rcnn_feature_map)
            cls_prob = F.softmax(cls_score)
            return bbox_pred, cls_score, cls_prob
        bbox_pred, cls_score, cls_prob = run_fast_rcnn()

        if self.training:
            faster_rcnn_loss_cls = F.cross_entropy(cls_score, rois_label)
            faster_rcnn_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        else:
            faster_rcnn_loss_cls = 0
            faster_rcnn_loss_bbox = 0

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, \
               rpn_loss_cls, rpn_loss_bbox, \
               faster_rcnn_loss_cls, faster_rcnn_loss_bbox, rois_label
