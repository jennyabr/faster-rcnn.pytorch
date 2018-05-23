from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch.nn as nn
import torchvision.models as models

from model.feature_extractors.faster_rcnn_feature_extractors import FasterRCNNFeatureExtractors
from model.utils.net_utils import assert_sequential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VGG16ForFasterRCNN(FasterRCNNFeatureExtractors):
    class _FastRCNNFeatureExtractor(nn.Module):
        def __init__(self, vgg_architecture, remove_last_layer_from_network_fn):
            super(VGG16ForFasterRCNN._FastRCNNFeatureExtractor, self).__init__()
            layers = remove_last_layer_from_network_fn(vgg_architecture.classifier)
            self.feature_extractor = nn.Sequential(*layers)

        def forward(self, input):
            flattened_input = input.view(input.size(0), -1)
            return self.feature_extractor(flattened_input)

    def __init__(self):
        super(VGG16ForFasterRCNN, self).__init__()

        def load_base(vgg):
            base_fe = nn.Sequential(*self._remove_last_layer_from_network(vgg.features))
            base_fe_non_trainable = self._freeze_layers(base_fe, 2)
            return base_fe_non_trainable

        def load_fast_rcnn(vgg):
            fast_rcnn_fe = self._FastRCNNFeatureExtractor(vgg, self._remove_last_layer_from_network)
            return fast_rcnn_fe

        vgg = models.vgg16()
        self._base_feature_extractor = load_base(vgg)
        self._fast_rcnn_feature_extractor = load_fast_rcnn(vgg)

    @property
    def base_feature_extractor(self):
        return self._base_feature_extractor

    @property
    def fast_rcnn_feature_extractor(self):
        return self._fast_rcnn_feature_extractor

    @property
    def base_subnet(self):
        return self._base_feature_extractor

    @property
    def fast_rcnn_subnet(self):
        return self._fast_rcnn_feature_extractor.feature_extractor

    def _remove_last_layer_from_network(self, model):
        return list(model._modules.values())[:-1]

    def _freeze_layers(self, model, upto_pooling_num):
        assert_sequential(model)
        curr_pooling_num = 0
        for idx in range(len(model)):
            layer = model[idx]
            layer_name = layer.__class__.__name__.lower()
            if layer_name.find('pool') != -1:
                curr_pooling_num += 1
            if curr_pooling_num >= upto_pooling_num:
                break

            for p in layer.parameters():
                p.requires_grad = False
        return model

    def get_output_num_channels(self, model):
        for layer_num in range(len(model) - 1, -1, -1):
            if hasattr(model[layer_num], 'out_channels'):
                return model[layer_num].out_channels
            if hasattr(model[layer_num], 'out_features'):
                return model[layer_num].out_features

        raise AssertionError('Unexpected model architecture')
