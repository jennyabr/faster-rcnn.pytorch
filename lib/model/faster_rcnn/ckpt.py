import logging
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_state_from_ckpt(trained_model_path, model, optimizer=None):
    def get_trainable_params(LR, WEIGHT_DECAY, BIAS_DECAY, DOUBLE_BIAS):
        trainable_params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    trainable_params += [{
                        'params': [value],
                        'lr': LR * (DOUBLE_BIAS + 1),
                        'weight_decay': BIAS_DECAY and WEIGHT_DECAY or 0}]
                else:
                    trainable_params += [{
                        'params': [value],
                        'lr': LR,
                        'weight_decay': WEIGHT_DECAY}]
                #TODO batchnorm has more parameters...
        return trainable_params

    logger.info("Loading state from {}.".format(trained_model_path))
    checkpoint = torch.load(trained_model_path)

    session = checkpoint['session']
    epoch = checkpoint['epoch']

    if not model:
        pass  # TODO should we load model? it is loaded in faster_rcnn obj or at least creat it

    if not optimizer:  # TODO
        MOMENTUM = checkpoint['MOMENTUM']  # TODO config?
        optim_type = checkpoint['optimizer_type']  # TODO config?
        trainable_params = get_trainable_params()  # TODO config?
        optimizer = optimizer_factory(optim_type, trainable_params, momentum=MOMENTUM)

    optimizer.load_state_dict(checkpoint['optimizer'])
    #lr = optimizer.param_groups[0]['lr']

    return session, epoch, model, optimizer, {}

def save_state_to_checkpoint(save_to, session, epoch, model, optimizer, cfg): #cfg.get_ckpt_path(epoch)
    if cfg.mGPUs:
        ckpt_model = model.module.state_dict()
    else:
        ckpt_model = model.state_dict()
    logger.info('Saving model checkpoint to {}.'.format(save_to))
    torch.save({'session': session,
                'epoch': epoch + 1,
                'model': ckpt_model,
                'optimizer': optimizer.state_dict(),
                'cfg': cfg}, save_to)

