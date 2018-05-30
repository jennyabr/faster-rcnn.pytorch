from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from abc import ABC, abstractmethod
import os.path

import torch

from model.utils.net_utils import normal_init


class FasterRCNNFeatureExtractors(ABC):

    def __init__(self, net_variant, frozen_blocks):
        self.net_variant = net_variant
        self.frozen_blocks = frozen_blocks

    @property
    @abstractmethod
    def base_feature_extractor(self):
        return NotImplementedError

    @property
    @abstractmethod
    def fast_rcnn_feature_extractor(self):
        return NotImplementedError

    @abstractmethod
    def recreate_state_dict(self, orig_state_dict):
        return NotImplementedError

    @abstractmethod
    def get_output_num_channels(self, model):
        raise NotImplementedError


def create_feature_extractor_empty(net_name, net_variant, frozen_blocks):
    fe_cls = _get_feature_extractor_cls(net_name)
    empty_fe = fe_cls(net_variant=net_variant, frozen_blocks=frozen_blocks)
    return empty_fe


def create_feature_extractor_with_random_normal_init(net_name, net_variant, mean=0, stddev=0.01):
    fe_cls = _get_feature_extractor_cls(net_name)
    fe = fe_cls(net_variant=net_variant, frozen_blocks=0)
    configured_normal_init = partial(normal_init, mean=mean, stddev=stddev)
    fe.base_feature_extractor.apply(configured_normal_init)
    fe.fast_rcnn_feature_extractor.feature_extractor.apply(configured_normal_init)
    return fe


def create_feature_extractor_from_ckpt(net_name, net_variant, frozen_blocks, pretrained_model_path):
    if pretrained_model_path is None or not os.path.exists(pretrained_model_path):
        raise ValueError('Pretrained model path given does not exist')
    fe_cls = _get_feature_extractor_cls(net_name)
    fe = fe_cls(net_variant=net_variant, frozen_blocks=frozen_blocks)
    orig_state_dict = torch.load(os.path.expanduser(pretrained_model_path))
    fe_subnets = fe.recreate_state_dict(orig_state_dict)
    for fe_subnet, new_state_dict in fe_subnets:
        fe_subnet.load_state_dict(new_state_dict, strict=False)
    return fe


def _get_feature_extractor_cls(net_name):
    from model.feature_extractors import feature_extractors_classes
    from importlib import import_module

    class_names = [c for c in feature_extractors_classes.keys() if c.lower().startswith(net_name.lower())]
    if len(class_names) == 0:
        raise ImportError('No feature extractor found matching the net_name: {}'.format(net_name))
    elif len(class_names) > 1:
        raise ImportError('net_name: {} is ambiguous, {} classes match this name.'.format(net_name, class_names))

    class_name = class_names[0]
    class_module = import_module(feature_extractors_classes[class_name])
    feature_extractor_cls = getattr(class_module, class_name)
    return feature_extractor_cls
