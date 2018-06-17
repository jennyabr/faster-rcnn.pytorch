import os

from pipeline.faster_rcnn.run_functions.run_classic_pipeline import create_and_train_with_err_handling
from utils.config import ConfigProvider
from utils.logging import set_root_logger

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config_file = os.path.join(os.getcwd(), 'cfgs', 'vgg16.yml')

    cfg = ConfigProvider()
    cfg.load(config_file)
    set_root_logger(cfg.get_log_path())

    create_and_train_with_err_handling(cfg)
