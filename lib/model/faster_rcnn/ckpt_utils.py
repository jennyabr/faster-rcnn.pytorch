import logging
import os

import torch

from cfgs.config import ConfigProvider
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_session_to_ckpt(model, optimizer, cfg, epoch):
    if cfg.mGPUs:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_to = cfg.get_ckpt_path(epoch)
    logger.info('Saving model checkpoint to {}.'.format(save_to))
    torch.save({'last_performed_epoch': epoch,
                'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'ckpt_cfg': cfg}, save_to)


def load_session_from_ckpt(ckpt_path):
    # TODO: JA - don't be hard coded to faster-rcnn (uses FasterRCNN constructor)
    model = FasterRCNNMetaArch.create_from_ckpt(ckpt_path)
    state_dict, loaded_cfg = load_ckpt_file(ckpt_path)

    def create_optimizer_from_ckpt_fn(trainable_params):
        optimizer = torch.optim.SGD(params=trainable_params, momentum=loaded_cfg.TRAIN.MOMENTUM)
        optimizer.load_state_dict(state_dict['optimizer'])
        return optimizer

    last_performed_epoch = state_dict['last_performed_epoch']
    return model, create_optimizer_from_ckpt_fn, loaded_cfg, last_performed_epoch


def load_ckpt_file(ckpt_path):
    state_dict = torch.load(os.path.expanduser(ckpt_path))
    loaded_cfg = ConfigProvider()
    loaded_cfg.load_from_dict(state_dict['ckpt_cfg'])
    return state_dict, loaded_cfg