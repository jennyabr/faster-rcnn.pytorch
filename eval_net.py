import argparse

from cfgs.config import cfg
from data_handler.detection_data_manager import FasterRCNNDataManager
from data_handler.data_manager_api import Mode
from model.faster_rcnn.faster_rcnn_evaluation import faster_rcnn_evaluation
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_postprocessing import faster_rcnn_postprocessing
from model.faster_rcnn.faster_rcnn_prediction import faster_rcnn_prediction
from model.faster_rcnn.faster_rcnn_visualization import faster_rcnn_visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--config_dir', dest='config_dir', help='Path to config dir', type=str)
    args = parser.parse_args()

    # TODO: IB: cfg should be a local variable to enable h.p. sweeps like the following.
    # TODO: IB it can be assigned to the state of the trainer\evaluator\etc.
    global cfg
    cfg.load(args.config_dir)
    predict_on_epoch = 6  # TODO: JA - enable this to get the value 'last'
    model = FasterRCNNMetaArch.create_from_ckpt(cfg.get_ckpt_path(predict_on_epoch))
    model.cuda()
    data_manager = FasterRCNNDataManager(mode=Mode.INFER,
                                              imdb_name=cfg.imdbval_name,
                                              seed=cfg.RNG_SEED,
                                              num_workers=cfg.NUM_WORKERS,
                                              is_cuda=cfg.CUDA,
                                              batch_size=cfg.TRAIN.batch_size)

    # faster_rcnn_prediction(data_manager, model, cfg, predict_on_epoch)
    #
    # faster_rcnn_postprocessing(data_manager, model, cfg, predict_on_epoch)
    #
    detections_path = cfg.get_postprocessed_detections_path(predict_on_epoch)
    eval_path = cfg.get_evals_dir_path(predict_on_epoch)
    faster_rcnn_evaluation(data_manager, cfg, detections_path, eval_path)

    faster_rcnn_visualization(data_manager, cfg, predict_on_epoch)

    print("fin")
