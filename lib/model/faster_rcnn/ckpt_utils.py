import os
import torch

from cfgs.config import ConfigProvider, get_logger


def save_session_to_ckpt(model, optimizer, cfg, epoch):
    logger = get_logger(__name__)
    if cfg.mGPUs:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_to = cfg.get_ckpt_path(epoch)
    logger.info(' --->>> Saving model checkpoint to: {}'.format(save_to))
    torch.save({'last_performed_epoch': epoch,
                'model': model_state_dict,
                'model_cfg_params': model.cfg_params,
                'optimizer': optimizer.state_dict(),
                'ckpt_cfg': cfg.get_state_dict()}, save_to)


def load_session_from_ckpt(ckpt_path):
    from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
    state_dict = torch.load(os.path.expanduser(ckpt_path))
    loaded_cfg = ConfigProvider()
    loaded_cfg.create_from_dict(state_dict['ckpt_cfg'])
    # TODO: JA - don't be hard coded to faster-rcnn (uses FasterRCNN constructor)
    model = FasterRCNNMetaArch.create_from_ckpt(ckpt_path)

    def create_optimizer_from_ckpt_fn(trainable_params):
        optimizer = torch.optim.SGD(params=trainable_params, momentum=loaded_cfg.TRAIN.MOMENTUM)
        optimizer.load_state_dict(state_dict['optimizer'])
        return optimizer

    last_performed_epoch = state_dict['last_performed_epoch']
    return model,create_optimizer_from_ckpt_fn, loaded_cfg, last_performed_epoch
