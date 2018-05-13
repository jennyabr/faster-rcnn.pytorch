from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod

import torch.nn as nn


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

    @abstractmethod
    def get_output_num_channels(self, model):
        raise NotImplementedError

    @staticmethod
    def check_sequential(model):
        if not isinstance(model, nn.Sequential):
            raise ValueError(
                'Since Pytorch supports dynamic graphs, the order of layers in a non-sequential net is '
                'defined only dynamically at run-time and cant be used for counting layers')
