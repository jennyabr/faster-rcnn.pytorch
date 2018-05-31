import torch

import os
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.feature_extractors.faster_rcnn_feature_extractors import create_feature_extractor_empty
from util.config import ConfigProvider

ckpt_path = ""  # TODO
ckpt_path_new = ""  # TODO

state_dict_new = torch.load(os.path.abspath(ckpt_path_new))
state_dict = torch.load(os.path.abspath(ckpt_path))
model_dict = state_dict['model']
new_model = ""  # TODO

loaded_cfg = ConfigProvider()
ckpt_cfg = {}  # TODO past from run
loaded_cfg.create_from_dict(ckpt_cfg)

feature_extractors = create_feature_extractor_empty("vgg", "16", 2)
model = FasterRCNNMetaArch(feature_extractors, loaded_cfg, 21)

model.load_state_dict(state_dict['model'])

# optimizer.load_state_dict(checkpoint['optimizer'])
# lr = optimizer.param_groups[0]['lr']
# optimizer = torch.optim.SGD(params=trainable_params, momentum=loaded_cfg.TRAIN.MOMENTUM)
# optimizer.load_state_dict(state_dict['optimizer'])

if 'pooling_mode' in state_dict.keys():
    POOLING_MODE = state_dict['pooling_mode']
