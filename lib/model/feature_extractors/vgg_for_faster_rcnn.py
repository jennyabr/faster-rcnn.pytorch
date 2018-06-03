from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
from torchvision.models import resnet101
from torchvision.models.resnet import resnet50, resnet152

from model.feature_extractors.faster_rcnn_feature_extractor_duo import FasterRCNNFeatureExtractorDuo
from model.utils.net_utils import global_average_pooling, remove_last_layer_from_network


class VGGForFasterRCNN(FasterRCNNFeatureExtractorDuo):

    def __init__(self, net_variant='16', frozen_blocks=0):
        super(VGGForFasterRCNN, self).__init__(net_variant, frozen_blocks)

        def vgg_variant_builder(variant):
            if str(variant) == '16':
                vgg = models.vgg16()
            else:
                raise ValueError('The variant VGG{} is not supported'.format(variant))
            return vgg
        vgg = vgg_variant_builder(net_variant)
        self._rpn_feature_extractor = self._RPNFeatureExtractor(vgg, frozen_blocks)
        self._fast_rcnn_feature_extractor = self._FastRCNNFeatureExtractor(vgg)

    @property
    def rpn_feature_extractor(self):
        return self._rpn_feature_extractor

    @property
    def fast_rcnn_feature_extractor(self):
        return self._fast_rcnn_feature_extractor
    
    class _RPNFeatureExtractor(FasterRCNNFeatureExtractorDuo._FeatureExtractor):
        def __init__(self, vgg, frozen_blocks):
            super(VGGForFasterRCNN._RPNFeatureExtractor, self).__init__()
            #TODO: JA - the model should not be able to change independently of the list of ordered layer names can change
            self._model = nn.Sequential(remove_last_layer_from_network(vgg.features))
            if not (0 <= frozen_blocks < 4):
                raise ValueError('Illegal number of blocks to freeze')
            self._frozen_blocks = frozen_blocks
            self._output_num_channels = self.get_output_num_channels(self._model[-1][-1].conv3) #TODO: JA - verify this

        @property
        def output_num_channels(self):
            return self._output_num_channels
        
        @property
        def layer_mapping_to_pretrained(self):
            return None

        def forward(self, input):
            return self._model(input)

        def train(self, mode=True):
            super(VGGForFasterRCNN._RPNFeatureExtractor, self).train(mode)
            VGGForFasterRCNN._freeze_layers(self._model, self._frozen_blocks)

    class _FastRCNNFeatureExtractor(FasterRCNNFeatureExtractorDuo._FeatureExtractor):

        def __init__(self, vgg):
            super(VGGForFasterRCNN._FastRCNNFeatureExtractor, self).__init__()
            layers = remove_last_layer_from_network(vgg.classifier)
            self._model = nn.Sequential(*layers)
            self._model.apply(self._freeze_batch_norm_layers)
            self._output_num_channels = self.get_output_num_channels(self._model[-1][-1].conv3) # TODO: verify this

        def forward(self, input):
            flattened_input = input.view(input.size(0), -1)
            return self._model(flattened_input)

        @property
        def output_num_channels(self):
            return self._output_num_channels
        
        @property
        def layer_mapping_to_pretrained(self):
            return None

        def train(self, mode=True):
            super(VGGForFasterRCNN._FastRCNNFeatureExtractor, self).train(mode)

    @classmethod
    def _freeze_layers(cls, model, upto_pooling_num):
        #TODO: JA - make sure this works correctly
        curr_pooling_num = 0
        for idx in range(len(model)):
            layer = model[idx]
            layer_name = layer.__class__.__name__.lower()
            if layer_name.find('pool') != -1:
                curr_pooling_num += 1
            if curr_pooling_num >= upto_pooling_num:
                break

            for p in layer.parameters():
                p.requires_grad = False
        return model

    def convert_pretrained_state_dict(self, pretrained_vgg_state_dict):
        rpn_state_dict = {}
        fast_rcnn_state_dict = {}

        for orig_key, v in pretrained_vgg_state_dict.items():
            if orig_key.startswith("features."):
                rpn_state_dict[orig_key.replace("features.", "_model.")] = v
            elif orig_key.startswith("classifier."):
                fast_rcnn_state_dict[orig_key.replace("classifier.", "_model.")] = v
            else:
                raise KeyError('unexpected key "{}" in state_dict'.format(orig_key))

        fe_subnets = [self.rpn_feature_extractor, self.fast_rcnn_feature_extractor]
        fe_state_dicts = [rpn_state_dict, fast_rcnn_state_dict]
        return zip(fe_subnets, fe_state_dicts)

