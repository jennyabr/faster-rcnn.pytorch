import os

from pipeline.faster_rcnn.run_functions.run_classic_pipeline import \
    create_and_train_with_err_handling, pred_eval_with_err_handling
from util.config import ConfigProvider
from util.logging import set_root_logger

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config_file_vgg16 = os.path.join(os.getcwd(), 'cfgs', 'vgg16.yml')
    config_file_resnet101 = os.path.join(os.getcwd(), 'cfgs', 'resnet101.yml')

    cfg_vgg16 = ConfigProvider()
    cfg_vgg16.load(config_file_vgg16)
    set_root_logger(cfg_vgg16.get_log_path())
    create_and_train_with_err_handling(cfg_vgg16)
    pred_eval_with_err_handling(cfg_vgg16)

    cfg_resnet101 = ConfigProvider()
    cfg_resnet101.load(config_file_resnet101)
    set_root_logger(cfg_resnet101.get_log_path())
    create_and_train_with_err_handling(cfg_resnet101)
    pred_eval_with_err_handling(cfg_resnet101)
