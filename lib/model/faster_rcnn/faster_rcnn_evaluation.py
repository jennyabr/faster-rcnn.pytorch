import time
import os
import logging
import numpy as np
import torch

from model.utils.net_utils import decay_lr_in_optimizer, clip_gradient, vis_detections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: IB - the path to the saved model dir should be passed to the init. The init should load the model
# TODO: IB - currently this is in test_model, and look for the config file in the saved model dir.
# TODO: IB - if the config file doesn't exist - it should raise an exception and not use the global config file
# TODO: IB - another alternative is to save all the configs together with the model weights in the same file,
# TODO: IB - instead of in a separate config file - if possible

def faster_rcnn_evaluation(data_manager, model, cfg):
    #TODO: Continue from here
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    end_time = time.time()
    print("test time: %0.4fs" % (end_time - start_time))
