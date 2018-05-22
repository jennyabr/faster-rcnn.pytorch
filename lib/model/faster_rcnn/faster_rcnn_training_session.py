import time
import logging

import torch
import torch.nn as nn
from model.utils.net_utils import decay_lr_in_optimizer, clip_gradient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training_session(data_manager, model, create_optimizer_fn, cfg, train_logger):
    def get_trainable_params():
        trainable_params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    trainable_params += [{'params': [value],
                                          'lr': cfg.TRAIN.LEARNING_RATE * (cfg.TRAIN.DOUBLE_BIAS + 1),
                                          'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    trainable_params += [{'params': [value],
                                          'lr': cfg.TRAIN.LEARNING_RATE,
                                          'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        return trainable_params
    trainable_params = get_trainable_params()
    optimizer = create_optimizer_fn(trainable_params)

    if cfg.mGPUs:
        model = nn.DataParallel(model)
    if cfg.CUDA:
        model.cuda()
    iters_per_epoch = data_manager.iters_per_epoch
    
    model.train()
    aggregated_stats = {}
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.max_epochs + 1):
        decay_lr_in_optimizer(epoch, cfg.TRAIN.lr_decay_step + 1, optimizer, cfg.TRAIN.GAMMA)

        epoch_start_time = time.time()
        aggregation_start_time = time.time()
        for step in range(iters_per_epoch):
            batch_outputs = _train_on_batch(data_manager, model, optimizer, cfg)
            aggregated_stats = aggregate_stats(aggregated_stats, batch_outputs, cfg.TRAIN.disp_interval)
            
            def log_training_stats():
                if step % cfg.TRAIN.disp_interval == 0:
                    aggregation_end_time = time.time()
                    time_per_sample = (aggregation_end_time - aggregation_start_time) / cfg.TRAIN.disp_interval 
                    _write_stats_to_logger(
                        train_logger=train_logger,
                        aggregated_stats=aggregated_stats,
                        time_per_sample=time_per_sample,
                        epoch=epoch, step=step, iters_per_epoch=iters_per_epoch)
            log_training_stats()
            aggregated_stats = {}
            aggregation_start_time = time.time()

        def save_checkpoint():
            if cfg.mGPUs:
                ckpt_model = model.module.state_dict()
            else:
                ckpt_model = model.state_dict()
            save_to = cfg.get_ckpt_path(epoch)
            logger.info('Saving model checkpoint to {}'.format(save_to))
            torch.save({'session': cfg.TRAIN.session,
                        'epoch': epoch + 1,
                        'model': ckpt_model,
                        'optimizer': optimizer.state_dict(),
                        'pooling_mode': cfg.POOLING_MODE,
                        'class_agnostic': cfg.TRAIN.class_agnostic}, save_to)
        save_checkpoint()

        epoch_end_time = time.time()
        logger.info("Finished epoch {} in {} ms".format(epoch, epoch_end_time - epoch_start_time))


def _train_on_batch(data_manager, model, optimizer, cfg):
    im_data, im_info, gt_boxes, num_boxes = next(data_manager)

    model.zero_grad()
    rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label = \
        model(im_data, im_info, gt_boxes, num_boxes)
    loss_tensors = {
        'rpn_cls': rpn_loss_cls.mean(),
        'rpn_bbox': rpn_loss_bbox.mean(),
        'RCNN_cls': RCNN_loss_cls.mean(),
        'RCNN_bbox':  RCNN_loss_bbox.mean()
        }
    loss_tensors['total'] = torch.cat(list(loss_tensors.values())).sum()

    batch_outputs = {
        'rois': rois,
        'rois_label': rois_label,
        'cls_prob': cls_prob,
        'bbox_pred': bbox_pred,
        'loss_tensors': loss_tensors
        }

    optimizer.zero_grad()
    loss_tensors['total'].backward()

    if cfg.CLIP_GRADIENTS:
        clip_gradient(model, cfg.clip_gradients)
    optimizer.step()

    return batch_outputs


def _write_stats_to_logger(train_logger, aggregated_stats, time_per_sample,
                           epoch, step, iters_per_epoch):

    agg_metrics = {}
    for loss_name, agg_loss_tensor in aggregated_stats['loss_tensors']:
        agg_metrics[loss_name] = agg_loss_tensor.data[0]
    rois_label = aggregated_stats['rois_label']
    agg_metrics['fg_cnt'] = torch.sum(rois_label.data.ne(0))
    agg_metrics['bg_cnt'] = rois_label.data.numel() - agg_metrics['fg_cnt']

    current_step = epoch * iters_per_epoch + step
    logged_string = "[epoch %2d][iter %4d/%4d], time per sample: %f \n" % \
                (epoch, step, iters_per_epoch, time_per_sample)

    for metric_name, metric_value in agg_metrics:
        train_logger.scalar_summary(metric_name, metric_value, current_step)
        logged_string += "\t\t\t{}: {}".format(metric_name, metric_value)


def aggregate_stats(aggregated_stats, new_stats, disp_interval):
    res = {}
    for stat_name, stat_value in new_stats.items():
        res[stat_name] = aggregated_stats.get(stat_name, 0) + new_stats[stat_name] / disp_interval # TODO: ib - make sure it happens on the gpu
    return res
