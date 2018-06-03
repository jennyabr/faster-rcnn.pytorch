from __future__ import absolute_import
from __future__ import division

import os.path
import torch
from torch import nn
from abc import ABC, abstractmethod
from functools import partial
from torch.nn.modules.batchnorm import _BatchNorm

from model.utils.net_utils import normal_init


class FeatureExtractorDuo(ABC):

    def __init__(self, net_variant, frozen_blocks):
        self.net_variant = net_variant
        self.frozen_blocks = frozen_blocks
        self._rpn_feature_extractor = None
        self._fast_rcnn_feature_extractor = None

    class _FeatureExtractor(nn.Module):
        @property
        def output_num_channels(self):
            raise NotImplementedError

        @classmethod
        def get_output_num_channels(cls, layer):
            if hasattr(layer, 'out_channels'):
                out_num = layer.out_channels
            elif hasattr(layer, 'out_features'):
                out_num = layer.out_features
            else:
                raise ValueError('Last layer of model does not have field describing number of output channels. '
                                 'Check if it was set correctly.')
            return out_num

        @classmethod
        def _freeze_batch_norm_layers(cls, module):
            def _freeze_if_batch_norm(sub_module):
                if isinstance(sub_module, _BatchNorm):
                    sub_module.eval()  # freezing running_mean & running_var
                    for p in sub_module.parameters():
                        p.requires_grad = False  # freezing weight & bias
            module.apply(_freeze_if_batch_norm)
        
        @property
        @abstractmethod
        def layer_mapping_to_pretrained(self):
            raise NotImplementedError

    @property
    @abstractmethod
    def rpn_feature_extractor(self):
        return NotImplementedError

    @property
    @abstractmethod
    def fast_rcnn_feature_extractor(self):
        return NotImplementedError

    @abstractmethod
    def convert_pretrained_state_dict(self, pretrained_state_dict):
        return NotImplementedError


def create_empty_duo(net_name, net_variant, frozen_blocks):
    fe_cls = _get_feature_extractor_duo_cls(net_name)
    empty_fe_duo = fe_cls(net_variant=net_variant, frozen_blocks=frozen_blocks)
    return empty_fe_duo


#TODO: JA - test this function
def create_duo_with_normal_init(net_name, net_variant, mean=0, stddev=0.01):
    fe_duo = create_empty_duo(net_name, net_variant, frozen_blocks=0)
    configured_normal_init = partial(normal_init, mean=mean, stddev=stddev)
    fe_duo.rpn_feature_extractor.apply(configured_normal_init)
    fe_duo.fast_rcnn_feature_extractor.apply(configured_normal_init)
    return fe_duo


def create_duo_from_ckpt(net_name, net_variant, frozen_blocks, pretrained_model_path):
    if pretrained_model_path is None or not os.path.exists(pretrained_model_path):
        raise ValueError('Pretrained model path given does not exist')
    fe_duo = create_empty_duo(net_name, net_variant, frozen_blocks=frozen_blocks)
    orig_state_dict = torch.load(os.path.abspath(pretrained_model_path))
    zipped_subnets_and_state_dicts = fe_duo.convert_pretrained_state_dict(orig_state_dict)
    for fe_subnet, new_state_dict in zipped_subnets_and_state_dicts:
        fe_subnet.load_state_dict(new_state_dict, strict=False)
    return fe_duo


def _get_feature_extractor_duo_cls(net_name):
    from model.feature_extractors import feature_extractors_duo_classes
    from importlib import import_module

    class_names = [c for c in feature_extractors_duo_classes.keys() if c.lower().startswith(net_name.lower())]
    if len(class_names) == 0:
        raise ImportError('No feature extractor duo found matching the net_name: {}'.format(net_name))
    elif len(class_names) > 1:
        raise ImportError('net_name: {} is ambiguous, {} classes match this name.'.format(net_name, class_names))

    class_name = class_names[0]
    class_module = import_module(feature_extractors_duo_classes[class_name])
    feature_extractor_duo_cls = getattr(class_module, class_name)
    return feature_extractor_duo_cls
