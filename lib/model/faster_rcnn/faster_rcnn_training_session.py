import logging
import time

import torch
import torch.nn as nn
from torch.autograd.variable import Variable

from model.faster_rcnn.ckpt_utils import save_session_to_ckpt
from model.utils.net_utils import decay_lr_in_optimizer, clip_gradient

logger = logging.getLogger(__name__)


def run_training_session(data_manager, model, create_optimizer_fn, cfg, train_logger, first_epoch=0):
    def get_trainable_params():
        trainable_params = []
        trainable_keys = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                trainable_keys += [key, ]
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

    aggregated_stats = {}
    model.train()
    for epoch in range(first_epoch, cfg.TRAIN.max_epochs + 1):
        decay_lr_in_optimizer(epoch, cfg.TRAIN.lr_decay_step + 1, optimizer, cfg.TRAIN.GAMMA)

        epoch_start_time = aggregation_start_time = time.time()
        data_manager.prepare_iter_for_new_epoch()
        for step in range(iters_per_epoch):
            batch_outputs = _train_on_batch(data_manager, model, optimizer, cfg)
            aggregated_stats = _aggregate_stats(aggregated_stats, batch_outputs['batch_metrics'], cfg.TRAIN.disp_interval)

            if step % cfg.TRAIN.disp_interval == 0 and step > 0:
                aggregation_end_time = time.time()
                time_per_sample = (aggregation_end_time - aggregation_start_time) / cfg.TRAIN.disp_interval
                logged_string = _write_stats_to_logger(
                    train_logger=train_logger,
                    metrics=aggregated_stats,
                    time_per_sample=time_per_sample,
                    epoch=epoch, step=step, iters_per_epoch=iters_per_epoch)
                logger.info(logged_string)
                aggregated_stats = {}
                aggregation_start_time = time.time()

        epoch_end_time = time.time()
        epoch_duration_hrs = (epoch_end_time - epoch_start_time) / 3600
        logger.info("Finished epoch {0} in {1:.3f} hrs.".format(epoch, epoch_duration_hrs))

        save_session_to_ckpt(model, optimizer, cfg, epoch)


def _train_on_batch(data_manager, model, optimizer, cfg):
    im_data, im_info, gt_boxes, num_boxes = next(data_manager)

    model.zero_grad()
    rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label = \
        model(im_data, im_info, gt_boxes, num_boxes)

    batch_metrics = {'loss_rpn_cls': rpn_loss_cls.mean(),
                    'loss_rpn_box': rpn_loss_bbox.mean(),
                    'loss_rcnn_cls': RCNN_loss_cls.mean(),
                    'loss_rcnn_box':  RCNN_loss_bbox.mean()}

    batch_metrics['loss'] = torch.cat(list(batch_metrics.values())).sum()

    # TODO variable?
    batch_metrics['fg_cnt'] = torch.sum(rois_label.data.ne(0))
    batch_metrics['bg_cnt'] = rois_label.data.numel() - batch_metrics['fg_cnt']

    batch_outputs = {
        'rois': rois,
        'rois_label': rois_label,
        'cls_prob': cls_prob,
        'bbox_pred': bbox_pred,
        'batch_metrics': batch_metrics
        }

    optimizer.zero_grad()
    batch_metrics['loss'].backward()

    if cfg.TRAIN.CLIP_GRADIENTS:
        clip_gradient(model, cfg.TRAIN.CLIP_GRADIENTS)
    optimizer.step()

    return batch_outputs


def _write_stats_to_logger(train_logger, metrics, time_per_sample, epoch, step, iters_per_epoch):
    current_step = epoch * iters_per_epoch + step
    logged_string = " [epoch {0}] [iter {1}/{2}]: time per sample: {3:.3f} min.".format(
        epoch, step, iters_per_epoch, time_per_sample)

    for metric_name, metric_value in metrics.items():
        train_logger.scalar_summary(metric_name, metric_value, current_step)
        logged_string += "\n\t\t{0}: {1:.4f}".format(metric_name, metric_value)

    return logged_string

def _aggregate_stats(aggregated_stats, new_stats, disp_interval):
    res = {}
    for stat_name, stat_value in new_stats.items():
        if type(stat_value) is Variable:
            # TODO: JA - can we aggregate the stats on the gpu instead of the cpu?
            current = stat_value.data[0]
        else:
            current = stat_value
        aggregated = aggregated_stats.get(stat_name, 0)
        res[stat_name] = aggregated + current / disp_interval
    return res
