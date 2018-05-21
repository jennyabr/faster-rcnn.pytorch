from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models

from abc import ABC, abstractmethod


class FasterRCNNFeatureExtractors(ABC):
    def __init__(self, pretrained, model_path):
        self.pretrained = pretrained
        self.model_path = model_path

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