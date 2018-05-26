import time
import os
import logging
import numpy as np
import torch
import torch.nn as nn

from model.feature_extractors.feature_extractors_factory import FeatureExtractorsFactory
from model.utils.net_utils import decay_lr_in_optimizer, clip_gradient
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.feature_extractors.resnet_for_faster_rcnn import ResNetForFasterRCNN
from model.feature_extractors.vgg_for_faster_rcnn import VGGForFasterRCNN
from data_handler.data_with_test import DataPrep, Mode

from cfgs.config import cfg
from roi_data_layer.roidb import combined_roidb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: IB - the path to the saved model dir should be passed to the init. The init should load the model
# TODO: IB - currently this is in test_model, and look for the config file in the saved model dir.
# TODO: IB - if the config file doesn't exist - it should raise an exception and not use the global config file
# TODO: IB - another alternative is to save all the configs together with the model weights in the same file,
# TODO: IB - instead of in a separate config file - if possible

def faster_rcnn_visualization(data_manager, model, cfg):
    start_time = time.time()

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)

            out_image_path = os.path.join(output_dir, 'result{}.png'.format(i))
            cv2.imwrite(out_image_path, im2show)






    end_time = time.time()
    print("test time: %0.4fs" % (end_time - start_time))