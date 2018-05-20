from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from functools import partial
from abc import ABC, abstractmethod
import os.path

import torch

from lib import FASTER_RCNN_LIB_FULL_PATH
from model.utils.net_utils import normal_init
from model.utils.python_utils import get_class_from_package


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

    @classmethod
    def create_with_random_normal_init(cls, mean=0, stddev=0.01):
        fe = cls()
        configured_normal_init = partial(normal_init, mean=mean, stddev=stddev)
        configured_normal_init(fe.base_feature_extractor)
        configured_normal_init(fe.fast_rcnn_feature_extractor)
        return fe

    @classmethod
    def create_from_ckpt(cls, pretrained_model_path):
        fe = cls()
        logger.info("Loading feature extractors pretrained weights from {}.".format(pretrained_model_path))
        state_dict = torch.load(os.path.expanduser(pretrained_model_path))
        fe_subnets = [fe.base_feature_extractor, fe.fast_rcnn_feature_extractor]
        for fe_subnet in fe_subnets:
            fe_subnet.load_state_dict({k: v for k, v in state_dict.items() if k in fe_subnet.state_dict()})
        return fe

    @abstractmethod
    def get_output_num_channels(self, model):
        raise NotImplementedError


def create_feature_extractor(net, pretrained_model_path=None, mean=0, stddev=0.01):
    abs_package_path = os.path.dirname(__file__)  # TODO check: __package__
    rel_package_path = \
        abs_package_path.replace(FASTER_RCNN_LIB_FULL_PATH, '').replace(os.path.sep, '.').strip('.')
    feature_extractor_class = \
        get_class_from_package(rel_package_path,
                               '{}_for_faster_rcnn.{}ForFasterRCNN'.format(net.lower(), net),
                               FasterRCNNFeatureExtractors)
    # TODO: IB - get_class_from_package should not be case sensitive
    # TODO       for the exact class\module name (e.g. vgg16 vs VGG16)

    if pretrained_model_path:
        fe = feature_extractor_class.create_from_ckpt(pretrained_model_path)
    else:
        fe = feature_extractor_class.create_with_random_normal_init(mean, stddev)

    return fe
