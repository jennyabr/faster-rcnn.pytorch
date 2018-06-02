import os

import torch

from functools import partial

from abc import ABC, abstractmethod
from torchvision.models import resnet50, resnet101, resnet152

from model.feature_extractors import FeatureExtractorDuoFactory
from model.feature_extractors.resnet_for_faster_rcnn import ResnetRPNFeatureExtractor, ResnetFastRCNNFeatureExtractor
from model.utils.net_utils import normal_init


class ResnetFeatureExtractorDuoFactory(FeatureExtractorDuoFactory):

    @classmethod
    def create_duo(cls, net_variant, frozen_blocks=0):
        def resnet_variant_builder(variant):
            if str(variant) == '50':
                return resnet50()
            elif str(variant) == '101':
                return resnet101()
            elif str(variant) == '152':
                return resnet152()
            else:
                raise ValueError('The variant Resnet{} is currently not supported'.format(variant))

        resnet = resnet_variant_builder(net_variant)
        rpn_fe = ResnetRPNFeatureExtractor(resnet, frozen_blocks=frozen_blocks)
        fast_rcnn_fe = ResnetFastRCNNFeatureExtractor(resnet)
        return rpn_fe, fast_rcnn_fe

    @classmethod
    def get_rpn_layer_mapping_to_pretrained(cls):
        ordered_layer_names = ["conv1.", "bn1.", "relu.", "maxpool.", "layer1.", "layer2.", "layer3."]
        mapping = {i: name for i, name in enumerate(ordered_layer_names)}
        return mapping

    @classmethod
    def get_fast_rcnn_layer_mapping_to_pretrained(cls):
        return {0: "layer4."}

    @classmethod
    def create_duo_from_ckpt(cls, net_variant, frozen_blocks, pretrained_model_path):
        if pretrained_model_path is None or not os.path.exists(pretrained_model_path):
            raise ValueError('Pretrained model path given does not exist')

        new_rpn_fe, new_fast_rcnn_fe = cls.create_duo(net_variant=net_variant)
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
    def convert_loaded_state_dict(cls, orig_state_dict, layer_mapping_dict):
        converted_state_dict = {}

        def startswith_one_of(queried_key, mapping_dict):
            for new_key, mapped_key in mapping_dict.items():
                if queried_key.startswith(mapped_key):
                    return new_key, mapped_key
            return None, None

        for orig_key, weights_val in orig_state_dict.items():
            new_key, string_to_replace = startswith_one_of(orig_key, layer_mapping_dict)
            if i != -1:
                converted_state_dict[orig_key.replace(string_to_replace, str(new_key) + ".")] = weights_val

        return converted_state_dict


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
