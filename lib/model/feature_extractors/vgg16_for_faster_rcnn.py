from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models

from abc import ABC, abstractmethod


class FasterRCNNFeatureExtractors(ABC):
    def __init__(self, pretrained):
        self.pretrained = pretrained

    @abstractmethod
    def get_base_feature_extractor(self):
        raise NotImplementedError

    @abstractmethod
    def get_fast_rcnn_feature_extractor(self):
        raise NotImplementedError

    @staticmethod
    def _remove_last_layer_from_network(subnet):
        return list(subnet._modules.values())[:-1]  # TODO???

    @staticmethod
    def _make_non_trainable(net, fixed_layers=10):  # TODO fixed_layers until pooling
        for layer in range(fixed_layers):
            for p in net[layer].parameters():
                p.requires_grad = False  # TODO make immutable
        return net

    @staticmethod
    def get_output_num_channels(model):
        for layer_num in range(len(model)-1, -1, -1):
            if hasattr(model[layer_num], 'out_channels'):
                return model[layer_num].out_channels
            if hasattr(model[layer_num], 'out_features'):
                return model[layer_num].out_features

        raise AssertionError('Unexpected model architecture')


class VGG16ForFasterRCNN(FasterRCNNFeatureExtractors):
    class _FastRCNNFeatureExtractor(nn.Module):
        def __init__(self, vgg_architecture):
            super(VGG16ForFasterRCNN._FastRCNNFeatureExtractor, self).__init__()
            layers = FasterRCNNFeatureExtractors._remove_last_layer_from_network(vgg_architecture.classifier)
            self.feature_extractor = nn.Sequential(*layers)

        def forward(self, input):
            flattened_input = input.view(input.size(0), -1)
            return self.feature_extractor(flattened_input)

    def __init__(self, pretrained=False, model_path='data/pretrained_model/vgg16_caffe.pth'):  # TODO defaults
        def load_vgg16():
            vgg = models.vgg16()
            if pretrained:
                # TODO log not print (how log effects the GPU?)
                print("Loading pretrained weights from %s" % model_path)
                state_dict = torch.load(model_path)
                vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
            return vgg

        def load_base(vgg):
            base_fe = nn.Sequential(*FasterRCNNFeatureExtractors._remove_last_layer_from_network(vgg.features))
            base_fe_non_trainable = self._make_non_trainable(base_fe, 10)
            return base_fe_non_trainable

        def load_fast_rcnn(vgg):
            fast_rcnn_fe = self._FastRCNNFeatureExtractor(vgg)
            return fast_rcnn_fe

        super(VGG16ForFasterRCNN, self).__init__(pretrained)
        self.model_path = model_path
        vgg = load_vgg16()
        self._base_feature_extractor = load_base(vgg)
        self._fast_rcnn_feature_extractor = load_fast_rcnn(vgg)

    def get_base_feature_extractor(self):
        return self._base_feature_extractor

    def get_fast_rcnn_feature_extractor(self):
        return self._fast_rcnn_feature_extractor
