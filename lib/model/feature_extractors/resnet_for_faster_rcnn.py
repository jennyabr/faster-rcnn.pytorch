from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torchvision.models import resnet101
from torchvision.models.resnet import Bottleneck
from model.utils.config import cfg
from model.feature_extractors.faster_rcnn_feature_extractors import FasterRCNNFeatureExtractors

class ResNetForFasterRCNN(FasterRCNNFeatureExtractors):
    class _FastRCNNFeatureExtractor(nn.Module):
        def __init__(self, resnet_architecture, freeze_batch_norm_layers_fn):
            super(ResNetForFasterRCNN._FastRCNNFeatureExtractor, self).__init__()
            self.feature_extractor = nn.Sequential(resnet_architecture.layer4)
            self.feature_extractor.apply(freeze_batch_norm_layers_fn)
            # TODO Is layer 4 can be frozen?

        def forward(self, input):
            return self.feature_extractor(input).mean(3).mean(2)

    def __init__(self, pretrained, model_path):
        super(ResNetForFasterRCNN, self).__init__(pretrained, model_path)

        def load_resnet():
            resnet = resnet101()  # TODO add more resnet types
            if self.pretrained:
                print("Loading pretrained weights from %s" % self.model_path)
                state_dict = torch.load(self.model_path)
                resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})
            return resnet

        def load_base(resnet):
            base_fe = nn.Sequential(resnet.conv1,
                                    resnet.bn1,
                                    resnet.relu,
                                    resnet.maxpool,
                                    resnet.layer1,
                                    resnet.layer2,
                                    resnet.layer3)
            assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)  # TODO move from here  ?
            base_fe_non_trainable = self._freeze_layers(base_fe, cfg.RESNET.FIXED_BLOCKS)
            base_fe.apply(self._freeze_batch_norm_layers)
            return base_fe_non_trainable

        def load_fast_rcnn(resnet):
            fast_rcnn_fe = self._FastRCNNFeatureExtractor(resnet, self._freeze_batch_norm_layers)
            return fast_rcnn_fe

        resnet = load_resnet()
        self._base_feature_extractor = load_base(resnet)
        self._fast_rcnn_feature_extractor = load_fast_rcnn(resnet)

    def _freeze_batch_norm_layers(self, net):
        def _freeze_if_batch_norm(model):
            if isinstance(model, nn.BatchNorm1d) or \
                    isinstance(model, nn.BatchNorm2d) or \
                    isinstance(model, nn.BatchNorm3d) or \
                    model.__class__.__name__.lower().find('batchnorm') != -1:
                model.eval()  # freezing running_mean & running_var
                for p in model.parameters():
                    p.requires_grad = False  # freezing weight & bias
        net.apply(_freeze_if_batch_norm)

    def _freeze_layers(self, model, upto_block_num):
        FasterRCNNFeatureExtractors.check_sequential(model)

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
        FasterRCNNFeatureExtractors.check_sequential(model)

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

    def get_base_feature_extractor(self):
        return self._base_feature_extractor

    def get_fast_rcnn_feature_extractor(self):
        return self._fast_rcnn_feature_extractor
