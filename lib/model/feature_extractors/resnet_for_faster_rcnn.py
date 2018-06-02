from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models import resnet101
from torchvision.models.resnet import Bottleneck, resnet50, resnet152

from model.feature_extractors.faster_rcnn_feature_extractors import FasterRCNNFeatureExtractors
from model.utils.net_utils import assert_sequential


class ResNetForFasterRCNN(FasterRCNNFeatureExtractors):

    def __init__(self, net_variant='101', frozen_blocks=0):
        super(ResNetForFasterRCNN, self).__init__(net_variant, frozen_blocks)

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
        self._rpn_feature_extractor = self._RPNFeatureExtractor(resnet, frozen_blocks)
        self._fast_rcnn_feature_extractor = self._FastRCNNFeatureExtractor(resnet)

    class _RPNFeatureExtractor(nn.Module):

        def __init__(self, resnet, frozen_blocks):
            super(ResNetForFasterRCNN._RPNFeatureExtractor, self).__init__()
            feature_extractor = nn.Sequential(resnet.conv1,
                                              resnet.bn1,
                                              resnet.relu,
                                              resnet.maxpool,
                                              resnet.layer1,
                                              resnet.layer2,
                                              resnet.layer3)
            if not (0 <= frozen_blocks < 4):
                raise ValueError('Illegal number of blocks to freeze')
            frozen_feature_extractor = ResNetForFasterRCNN._freeze_layers(feature_extractor, frozen_blocks)
            frozen_feature_extractor.apply(self._freeze_batch_norm_layers)
            self.model = frozen_feature_extractor

        def forward(self, input):
            return self.model(input)

    class _FastRCNNFeatureExtractor(nn.Module):

        def __init__(self, resnet):
            super(ResNetForFasterRCNN._FastRCNNFeatureExtractor, self).__init__()
            self.model = nn.Sequential(resnet.layer4)
            self.model.apply(ResNetForFasterRCNN._freeze_batch_norm_layers)

        def forward(self, input):
            def global_average_pooling(first_input):
                return first_input.mean(3).mean(2)
            result_feature_map = self.model(input)
            pooled_feature_vector = global_average_pooling(result_feature_map)
            return pooled_feature_vector


    @property
    def rpn_feature_extractor(self):
        return self._rpn_feature_extractor

    @property
    def fast_rcnn_feature_extractor(self):
        return self._fast_rcnn_feature_extractor

    @classmethod
    def _freeze_batch_norm_layers(cls, net):
        def _freeze_if_batch_norm(model):
            if isinstance(model, _BatchNorm):
                model.eval()  # freezing running_mean & running_var
                for p in model.parameters():
                    p.requires_grad = False  # freezing weight & bias
        net.apply(_freeze_if_batch_norm)

    @classmethod
    def _freeze_layers(cls, model, upto_block_num):
        curr_block_num = 0
        if upto_block_num > 0:
            for module in model.modules():
                module_name = module.__class__.__name__.lower()
                isBlock = module_name.find('layer') != -1
                if module_name.find('pool') != -1 or isBlock:
                    curr_block_num += 1
                if curr_block_num > upto_block_num:
                    return
                else:
                    if isBlock:
                        for submodule in module.modules():
                            for p in submodule.parameters():
                                p.requires_grad = False
                    else:
                        for p in module.parameters():
                            p.requires_grad = False
        return model

    def get_output_num_channels(self, model):
        '''
        this method supports only canonical Sequential arch
        :param model:
        :return:
        '''
        assert_sequential(model)

        def _get_output_num(layer):
            if hasattr(layer, 'out_channels'):
                out_num = layer.out_channels
            elif hasattr(layer, 'out_features'):
                out_num = layer.out_features
            else:
                out_num = None
            return out_num

        def _get_output_num_channels(model):
            for layer_num in range(len(model) - 1, -1, -1):
                if isinstance(model[layer_num], nn.Sequential):
                    out_num = _get_output_num_channels(model[layer_num])
                elif isinstance(model[layer_num], Bottleneck):
                    if model[layer_num].downsample:
                        out_num = _get_output_num_channels(model[layer_num].downsample)
                    else:
                        out_num = _get_output_num(model[layer_num].conv3)
                        if not out_num:
                            out_num = _get_output_num(model[layer_num].conv2)
                        if not out_num:
                            out_num = _get_output_num(model[layer_num].conv1)
                else:
                    out_num = _get_output_num(model[layer_num])

                if out_num:
                    return out_num
            return None

        out_num = _get_output_num_channels(model)
        if out_num:
            return out_num
        else:
            raise AssertionError('Unexpected model architecture')

    def recreate_state_dict(self, resnet_state_dict):
        base_state_dict = fast_rcnn_state_dict = {}

        base_layers_mapping = ["conv1.", "bn1.", "relu.", "maxpool.", "layer1.", "layer2.", "layer3."]
        fast_rcnn_layers_mapping = ["layer4."]

        def startswith_one_of(key, list):
            for i, item in enumerate(list):
                if key.startswith(item):
                    return i, item
            return -1, ""

        for orig_key, v in resnet_state_dict.items():
            i, item = startswith_one_of(orig_key, fast_rcnn_layers_mapping)
            if i != -1:
                fast_rcnn_state_dict[orig_key.replace(item, str(i) + ".")] = v
            else:
                i, item = startswith_one_of(orig_key, base_layers_mapping)
                if i != -1:
                    base_state_dict[orig_key.replace(item, str(i) + ".")] = v

        return [(self._base_feature_extractor, base_state_dict),
                (self._fast_rcnn_feature_extractor.feature_extractor, fast_rcnn_state_dict)]

    #TODO: JA - this was taken as is from the original repo, some of the lines look redundant\wrong
    def train(self, mode=True):
        nn.Module.train(self._base_feature_extractor, mode)
        nn.Module.train(self._fast_rcnn_feature_extractor, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self._base_feature_extractor.eval()
            self._base_feature_extractor[5].train()
            self._base_feature_extractor[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self._base_feature_extractor.apply(set_bn_eval)
            self._fast_rcnn_feature_extractor.apply(set_bn_eval)
