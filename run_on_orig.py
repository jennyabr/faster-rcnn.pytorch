import os
import torch
from model.faster_rcnn.faster_rcnn_meta_arch import FasterRCNNMetaArch
from model.feature_extractors.faster_rcnn_feature_extractors import create_feature_extractor_empty
from util.config import ConfigProvider

cfg_dict = {
 'ANCHOR_RATIOS': [0.5, 1, 2],
 'ANCHOR_SCALES': [8, 16, 32],
 'CROP_RESIZE_WITH_MAX_POOL': False,
 'CUDA': True,
 'DATA_DIR': '/home/jenny/gripper2/data',
 'DEDUP_BOXES': 0.0625,
 'DEDUP_BOXES_denominator': 16.0,
 'DEDUP_BOXES_numerator': 1.0,
 'EPS': 1e-14,
 'EXPERIMENT_NAME': 'only_numpy_seed3',
 'EXP_DIR': 'vgg16',
 'FEAT_STRIDE': [16],
 'GPU_ID': 0,
 'MAX_NUM_GT_BOXES': 20,
 'NUM_WORKERS': 6,
 'OUTPUT_DIR': '/home/jenny/gripper2/outputs/05_31',
 'OUTPUT_PATH': '/home/jenny/gripper2/outputs/05_31/only_numpy_seed3',
 'PIXEL_MEANS': [[[102.9801, 115.9465, 122.7717]]], #TODO was with array
 'POOLING_SIZE': 7,
 'RNG_SEED': 3,
 'TEST': {'DETECTION_THRESH': 0.05,
  'HAS_RPN': True,
  'MAX_DETECTIONS_PER_IMAGE': 100,
  'MAX_SIZE': 1000,
  'MODE': 'nms',
  'NMS': 0.3,
  'PROPOSAL_METHOD': 'gt',
  'RPN_MIN_SIZE': 16,
  'RPN_NMS_THRESH': 0.7,
  'RPN_POST_NMS_TOP_N': 300,
  'RPN_PRE_NMS_TOP_N': 6000,
  'RPN_TOP_N': 5000,
  'SCALES': [600],
  'USE_FLIPPED': False,
  'max_per_image': 100},
 'TRAIN': {'ASPECT_GROUPING': False,
  'BATCH_SIZE': 256,
  'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
  'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
  'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],
  'BBOX_NORMALIZE_TARGETS': True,
  'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
  'BBOX_REG': True,
  'BBOX_THRESH': 0.5,
  'BG_THRESH_HI': 0.5,
  'BG_THRESH_LO': 0.0,
  'BIAS_DECAY': False,
  'BN_TRAIN': False,
  'CLIP_GRADIENTS': 10.0,
  'DISPLAY': 10,
  'DOUBLE_BIAS': True,
  'FG_FRACTION': 0.25,
  'FG_THRESH': 0.5,
  'GAMMA': 0.1,
  'HAS_RPN': True,
  'IMS_PER_BATCH': 1,
  'LEARNING_RATE': 0.001,
  'MAX_SIZE': 1000,
  'MOMENTUM': 0.9,
  'PROPOSAL_METHOD': 'gt',
  'RPN_BATCHSIZE': 256,
  'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
  'RPN_CLOBBER_POSITIVES': False,
  'RPN_FG_FRACTION': 0.5,
  'RPN_MIN_SIZE': 8,
  'RPN_NEGATIVE_OVERLAP': 0.3,
  'RPN_NMS_THRESH': 0.7,
  'RPN_POSITIVE_OVERLAP': 0.7,
  'RPN_POSITIVE_WEIGHT': -1.0,
  'RPN_POST_NMS_TOP_N': 2000,
  'RPN_PRE_NMS_TOP_N': 12000,
  'SCALES': [600],
  'SNAPSHOT_ITERS': 5000,
  'SNAPSHOT_KEPT': 3,
  'SNAPSHOT_PREFIX': 'res101_faster_rcnn',
  'STEPSIZE': [30000],
  'SUMMARY_INTERVAL': 180,
  'TRIM_HEIGHT': 600,
  'TRIM_WIDTH': 600,
  'USE_ALL_GT': True,
  'USE_FLIPPED': True,
  'USE_GT': False,
  'WEIGHT_DECAY': 0.0005,
  'batch_size': 1,
  'checkepoch': 1,
  'checkpoint': 0,
  'checkpoint_interval': 10000,
  'checksession': 1,
  'disp_interval': 100,
  'frozen_blocks': 2,
  'large_scale': False,
  'lr_decay_step': 5,
  'max_epochs': 6,
  'optimizer': 'sgd',
  'pretrained_model_path': '/home/jenny/gripper2/data/pretrained_model/vgg16_caffe.pth',
  'resume': False,
  'session': 1,
  'start_epoch': 1,
  'use_tfboard': True},
 'USE_GPU_NMS': True,
 'ckpt_file_format': 'ckpt_e{}.pth',
 'class_agnostic': False,
 'coco': {'ANCHOR_RATIOS': [0.5, 1, 2],
  'ANCHOR_SCALES': [4, 8, 16, 32],
  'MAX_NUM_GT_BOXES': 50,
  'imdb_name': 'coco_2014_train+coco_2014_valminusminival',
  'imdbval_name': 'coco_2014_minival'},
 'dataset': 'pascal_voc',
 'evals_dir_format': 'evals_e{}',
 'imdb_name': 'voc_2007_trainval',
 'imdbval_name': 'voc_2007_test',
 'mGPUs': False,
 'net': 'vgg',
 'net_variant': '16',
 'num_regression_outputs_per_bbox': 4,
 'pascal_voc': {'ANCHOR_RATIOS': [0.5, 1, 2],
  'ANCHOR_SCALES': [8, 16, 32],
  'MAX_NUM_GT_BOXES': 20,
  'imdb_name': 'voc_2007_trainval',
  'imdbval_name': 'voc_2007_test'},
 'pascal_voc_0712': {'ANCHOR_RATIOS': [0.5, 1, 2],
  'ANCHOR_SCALES': [8, 16, 32],
  'MAX_NUM_GT_BOXES': 20,
  'imdb_name': 'voc_2007_trainval+voc_2012_trainval',
  'imdbval_name': 'voc_2007_test'},
 'postprocessed_file_format': 'pp_preds_e{}.pkl',
 'raw_preds_file_format': 'raw_preds_e{}.pkl',
 'roi_pooler_name': 'align',
 'roi_pooler_size': 7,
 'start_run_time_str': '2018_Jun_01_05_06',
 'vis_path_format': 'visualizations_e{}/{}.png'}

cfg = ConfigProvider()
cfg.create_from_dict(cfg_dict)

orig_ckpt_path = "/home/jenny/gripper2/outputs/orig/1/faster-rcnn.pytorch_orig_vgg16_pascal_voc/faster_rcnn_model_1_6_10021.pth"
orig_state_dict = torch.load(os.path.abspath(orig_ckpt_path))

ckpt_path = "/home/jenny/gripper2/outputs/05_31/after_config/ckpt_e6.pth"
state_dict = torch.load(os.path.abspath(ckpt_path))
feature_extractors = create_feature_extractor_empty('vgg', 16, 2)
model = FasterRCNNMetaArch(feature_extractors, cfg, 21)

model_state_dict = {}  #TODO

model.load_state_dict(model_state_dict)

print(1)
