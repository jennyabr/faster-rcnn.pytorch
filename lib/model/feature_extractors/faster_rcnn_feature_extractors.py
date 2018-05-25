from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from functools import partial
from abc import ABC, abstractmethod
import os.path

import torch

from model.utils.net_utils import normal_init


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FasterRCNNFeatureExtractors(ABC):

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

    @classmethod
    def create_with_random_normal_init(cls, mean=0, stddev=0.01):
        fe = cls(0)
        configured_normal_init = partial(normal_init, mean=mean, stddev=stddev)
        # configured_normal_init(fe.base_feature_extractor)
        # configured_normal_init(fe.fast_rcnn_feature_extractor.feature_extractor) #TODO should this be func?

        fe.base_feature_extractor.apply(configured_normal_init)
        fe.fast_rcnn_feature_extractor.feature_extractor.apply(configured_normal_init)

        return fe

    @classmethod
    def create_from_ckpt(cls, freeze, pretrained_model_path):
        fe = cls(freeze)
        logger.info(" Loading feature extractors pretrained weights from: {}.".format(pretrained_model_path))
        orig_state_dict = torch.load(os.path.expanduser(pretrained_model_path))
        fe_subnets = fe.recreate_state_dict(orig_state_dict)
        for fe_subnet, new_state_dict in fe_subnets:
            fe_subnet.load_state_dict(new_state_dict, strict=False)

        return fe


def create_feature_extractor(net, freeze=0, pretrained_model_path=None, mean=0, stddev=0.01):
    #TODO code reviwe

    from model.feature_extractors import feature_extractors_classes
    from importlib import import_module

    class_names = [c for c in feature_extractors_classes.keys() if c.lower().startswith(net)]
    if len(class_names) == 0:
        raise ImportError('{} is not part of the package!'.format(net))  # TODO msg
    elif len(class_names) > 1:
        raise ImportError('{} is umbiguas {}'.format(net, class_names))  # TODO msg
    else:
        class_name = class_names[0]
        try:
            class_module = import_module(feature_extractors_classes[class_name])
            feature_extractor_class = getattr(class_module, class_name)
        except (AttributeError, ModuleNotFoundError):
            raise ImportError('{} is not part of the package!'.format(class_name))

        if pretrained_model_path:
            fe = feature_extractor_class.create_from_ckpt(freeze=freeze, pretrained_model_path=pretrained_model_path)
        else:
            fe = feature_extractor_class.create_with_random_normal_init(mean, stddev)

        return fe

