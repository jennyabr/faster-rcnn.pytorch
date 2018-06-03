from __future__ import absolute_import
from __future__ import division
import os

from pipeline.faster_rcnn.run_functions.run_classic_pipeline import create_and_train, pred_eval

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from util.config import ConfigProvider


config_file = os.path.join(os.getcwd(), 'demos', 'cfgs', 'vgg16.yml')

cfg = ConfigProvider()
cfg.load(config_file)
create_and_train(cfg)
pred_eval(cfg)

except Exception:
    logger.error("Unexpected error: ", exc_info=True)
