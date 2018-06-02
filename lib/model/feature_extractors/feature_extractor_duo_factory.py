import torch

from functools import partial
import os
from abc import ABC, abstractmethod

from model.utils.net_utils import normal_init


class FeatureExtractorDuoFactory(metaclass=ABC):

    @classmethod
    @abstractmethod
    def create_duo(cls, net_variant, frozen_blocks=0):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_rpn_layer_mapping_to_pretrained(cls):
        return NotImplementedError

    @classmethod
    @abstractmethod
    def get_fast_rcnn_layer_mapping_to_pretrained(cls):
        return NotImplementedError

    @classmethod
    def create_duo_from_ckpt(cls, net_variant, frozen_blocks, pretrained_model_path):
        if pretrained_model_path is None or not os.path.exists(pretrained_model_path):
            raise ValueError('Pretrained model path given does not exist')
        new_rpn_fe, new_fast_rcnn_fe = \
            cls.create_duo(net_variant=net_variant, frozen_blocks=frozen_blocks)
        orig_state_dict = torch.load(os.path.abspath(pretrained_model_path))

        rpn_state_dict = cls.convert_loaded_state_dict(
            orig_state_dict, cls.get_rpn_layer_mapping_to_pretrained())
        new_rpn_fe.load_state_dict(rpn_state_dict, strict=False)

        fast_rcnn_state_dict = cls.convert_loaded_state_dict(
            orig_state_dict, cls.get_fast_rcnn_layer_mapping_to_pretrained())
        new_fast_rcnn_fe.load_state_dict(fast_rcnn_state_dict, strict=False)

        return new_rpn_fre, new_fast_rcnn_fe

    @classmethod
    @abstractmethod
    def convert_loaded_state_dict(cls, orig_state_dict, layer_mapping):
        raise NotImplementedError


def get_feature_extractor_duo_factory(net_name):
    from model.feature_extractors import feature_extractor_duo_factory_classes as duo_factories
    from importlib import import_module

    class_names = [c for c in duo_factories.keys() if c.lower().startswith(net_name.lower())]
    if len(class_names) == 0:
        raise ImportError('No duo factory found matching the net_name: {}'.format(net_name))
    elif len(class_names) > 1:
        raise ImportError('net_name: {} is ambiguous, {} classes match this name.'.format(net_name, class_names))

    class_name = class_names[0]
    class_module = import_module(duo_factories[class_name])
    duo_factory_cls = getattr(class_module, class_name)
    return duo_factory_cls
