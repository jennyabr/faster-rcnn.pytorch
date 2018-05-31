from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import torchvision.models as models

from model.feature_extractors.faster_rcnn_feature_extractors import FasterRCNNFeatureExtractors
from model.utils.net_utils import assert_sequential


class VGGForFasterRCNN(FasterRCNNFeatureExtractors):
    class _FastRCNNFeatureExtractor(nn.Module):
        def __init__(self, vgg_architecture, remove_last_layer_from_network_fn):
            super(VGGForFasterRCNN._FastRCNNFeatureExtractor, self).__init__()
            layers = remove_last_layer_from_network_fn(vgg_architecture.classifier)
            self.feature_extractor = nn.Sequential(*layers)

        def forward(self, input):
            flattened_input = input.view(input.size(0), -1)
            return self.feature_extractor(flattened_input)

    def __init__(self, net_variant='16', frozen_blocks=2):
        super(VGGForFasterRCNN, self).__init__(net_variant, frozen_blocks)

        def load_base(vgg):
            base_fe = nn.Sequential(*self._remove_last_layer_from_network(vgg.features))
            base_fe_non_trainable = self._freeze_layers(base_fe, frozen_blocks)
            return base_fe_non_trainable

        def load_fast_rcnn(vgg):
            fast_rcnn_fe = self._FastRCNNFeatureExtractor(vgg, self._remove_last_layer_from_network)
            return fast_rcnn_fe

        if net_variant != '16':
            raise ValueError('Currently only VGG16 is supported')
        vgg = models.vgg16()
        self._base_feature_extractor = load_base(vgg)
        self._fast_rcnn_feature_extractor = load_fast_rcnn(vgg)

    @property
    def base_feature_extractor(self):
        return self._base_feature_extractor

    @property
    def fast_rcnn_feature_extractor(self):
        return self._fast_rcnn_feature_extractor

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

    def recreate_state_dict(self, vgg16_state_dict):
        base_state_dict = {}
        fast_rcnn_state_dict = {}

        for orig_key, v in vgg16_state_dict.items():
            if orig_key.startswith("features."):
                base_state_dict[orig_key.replace("features.", "")] = v
            elif orig_key.startswith("classifier."):
                fast_rcnn_state_dict[orig_key.replace("classifier.", "")] = v
            else:
                raise KeyError('unexpected key "{}" in state_dict'.format(orig_key))

        return [(self._base_feature_extractor, base_state_dict),
                (self._fast_rcnn_feature_extractor.feature_extractor, fast_rcnn_state_dict)]
