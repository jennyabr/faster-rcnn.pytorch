from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
from torchvision.models import resnet101
from torchvision.models.resnet import resnet50, resnet152

from model.feature_extractors.faster_rcnn_feature_extractor_duo import FasterRCNNFeatureExtractorDuo


class ResNetForFasterRCNN(FasterRCNNFeatureExtractorDuo):

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

    @property
    def rpn_feature_extractor(self):
        return self._rpn_feature_extractor

    @property
    def fast_rcnn_feature_extractor(self):
        return self._fast_rcnn_feature_extractor
    
    class _RPNFeatureExtractor(FasterRCNNFeatureExtractorDuo._FeatureExtractor):

        def __init__(self, resnet, frozen_blocks):
            super(ResNetForFasterRCNN._RPNFeatureExtractor, self).__init__()
            self._ordered_layer_names = ["conv1.", "bn1.", "relu.", "maxpool.", "layer1.", "layer2.", "layer3."] 
            #TODO: JA - the model should not be able to change independently of the list of ordered layer names can change 
            self._model = nn.Sequential(resnet.conv1,
                                              resnet.bn1,
                                              resnet.relu,
                                              resnet.maxpool,
                                              resnet.layer1,
                                              resnet.layer2,
                                              resnet.layer3)
            if not (0 <= frozen_blocks < 4):
                raise ValueError('Illegal number of blocks to freeze')
            ResNetForFasterRCNN._freeze_layers(self._model, frozen_blocks)
            self._model.apply(self._freeze_batch_norm_layers)
            self._output_num_channels = self.get_output_num_channels(self._model[-1][-1].conv3)
            #TODO: JA - verify conv3

        @property
        def output_num_channels(self):
            return self._output_num_channels
        
        @property
        def layer_mapping_to_pretrained(self):
            mapping_dict = {layer_ind: layer_name for layer_ind, layer_name in enumerate(self._ordered_layer_names)}
            return mapping_dict

        def forward(self, input):
            return self._model(input)

        def train(self, mode=True):
            # TODO: JA - this was taken as is from the original repo, some of the lines look redundant\wrong
            nn.Module.train(self, mode)
            if mode:
                # TODO: JA - why 5, and 6 are in train? Should it be related to the frozen layers
                self.eval()
                self._model[5].train()
                self._model[6].train()
                
            # TODO: JA - use jenny's function and put it in the Base class
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self._model.apply(set_bn_eval)

    class _FastRCNNFeatureExtractor(FasterRCNNFeatureExtractorDuo._FeatureExtractor):

        def __init__(self, resnet):
            super(ResNetForFasterRCNN._FastRCNNFeatureExtractor, self).__init__()
            self._mapping_dict = {0: "layer4."}
            #TODO: JA - the model should not be able to change independently of the list of ordered layer names can change 
            self._model = nn.Sequential(resnet.layer4)

            self._model.apply(self._freeze_batch_norm_layers)
            self._output_num_channels = self.get_output_num_channels(self._model[-1][-1].conv3)

        def forward(self, input):
            return self._model(input).mean(3).mean(2)

        @property
        def output_num_channels(self):
            return self._output_num_channels
        
        @property
        def layer_mapping_to_pretrained(self):
            return self._mapping_dict

        def train(self, mode=True):
            nn.Module.train(self, mode)

            # TODO: JA - use jenny's function and put it in the Base class
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self._model.apply(set_bn_eval)

    @classmethod
    def _freeze_layers(cls, model, upto_block_num):
        #TODO: JA - use recursion instead of module.modules()
        curr_block_num = 0
        if upto_block_num > 0:
            for module in model.children():
                module_name = module.__class__.__name__.lower()
                is_block = isinstance(module, nn.Sequential)
                if module_name.find('pool') != -1 or is_block:
                    curr_block_num += 1
                if curr_block_num > upto_block_num:
                    break
                else:
                    if is_block:
                        for submodule in module.modules():
                            for p in submodule.parameters():
                                p.requires_grad = False
                    else:
                        for p in module.parameters():
                            p.requires_grad = False

    def convert_pretrained_state_dict(self, pretrained_resnet_state_dict):
        rpn_state_dict = {}
        fast_rcnn_state_dict = {}

        def startswith_one_of(key, mapping_dict):
            for i, item in mapping_dict.items():
                if key.startswith(item):
                    print('key: {}, item: {}, i: {}'.format(key, item, i)) #TODO: JA - delete this
                    return i, item
            return -1, ""

        for orig_key, v in pretrained_resnet_state_dict.items():
            i, item = startswith_one_of(orig_key, self.fast_rcnn_feature_extractor.layer_mapping_to_pretrained)
            if i != -1:
                fast_rcnn_state_dict[orig_key.replace(item, '_model.{}.'.format(str(i)))] = v
            else:
                i, item = startswith_one_of(orig_key, self.rpn_feature_extractor.layer_mapping_to_pretrained)
                if i != -1:
                    rpn_state_dict[orig_key.replace(item, '_model.{}.'.format(str(i)))] = v

        fe_subnets = [self.rpn_feature_extractor, self.fast_rcnn_feature_extractor]
        fe_state_dicts = [rpn_state_dict, fast_rcnn_state_dict]
        return zip(fe_subnets, fe_state_dicts)
