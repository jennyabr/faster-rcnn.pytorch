import argparse

from cfgs.config import cfg
from data_handler.detection_data_manager import FasterRCNNDataManager
from data_handler.data_manager_api import Mode
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.faster_rcnn.faster_rcnn_prediction import faster_rcnn_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--config_dir', dest='config_dir', help='Path to config dir', type=str)
    args = parser.parse_args()

    # TODO: IB: cfg should be a local variable to enable h.p. sweeps like the following.
    # TODO: IB it can be assigned to the state of the trainer\evaluator\etc.
    global cfg
    cfg.load(args.config_dir)

    eval_data_manager = FasterRCNNDataManager(mode=Mode.INFER,
                                              imdb_name=cfg.imdbval_name,
                                              seed=cfg.RNG_SEED,
                                              num_workers=cfg.NUM_WORKERS,
                                              is_cuda=cfg.CUDA,
                                              batch_size=cfg.TRAIN.batch_size)

    faster_rcnn_prediction(eval_data_manager, model, cfg)

    print("fin")
