from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import logging
import pprint
import torch
import yaml
from easydict import EasyDict as edict, EasyDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigProvider(dict):#object):
    def __init__(self):
        self._cfg = edict({})

    def load(self, config_dir_path):
        """Load a config file and merge it into the default options."""

        if config_dir_path:
            config_path = config_dir_path
        else:
            config_path = os.path.dirname(__file__)
        cfg = {}
        with open(os.path.join(config_path, 'general.yml'), 'r') as f:
            cfg = yaml.load(f)

        model_name = "{}_ls.yml".format(cfg.net) if cfg['TRAIN']['large_scale'] else "{}.yml".format(cfg['net'])
        with open(os.path.join(config_path, model_name), 'r') as f:
            model_cfg = yaml.load(f)

        if model_cfg:
            for k, v in model_cfg.items():
                if k == 'TRAIN' or k == 'TEST': #TODO can ask if len > 1
                    for k1, v1 in v.items():  # TODO note that this is not recursion...
                        cfg[k][k1] = v1
                else:
                    cfg[k] = v

        cfg['ROOT_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cfg['DATA_DIR'] = os.path.abspath(cfg['DATA_DIR'])

        if cfg['dataset'] == "pascal_voc":
            cfg['imdb_name'] = "voc_2007_trainval"
            cfg['imdbval_name'] = "voc_2007_test"
            cfg['ANCHOR_SCALES'] = [8, 16, 32]
            cfg['ANCHOR_RATIOS'] = [0.5, 1, 2]
            cfg['MAX_NUM_GT_BOXES'] = 20
        elif cfg['dataset'] == "pascal_voc_0712":
            cfg['imdb_name'] = "voc_2007_trainval+voc_2012_trainval"
            cfg['imdbval_name'] = "voc_2007_test"
            cfg['ANCHOR_SCALES'] = [8, 16, 32]
            cfg['ANCHOR_RATIOS'] = [0.5, 1, 2]
            cfg['MAX_NUM_GT_BOXES'] = 20
        elif cfg['dataset'] == "coco":
            cfg['imdb_name'] = "coco_2014_train+coco_2014_valminusminival"
            cfg['imdbval_name'] = "coco_2014_minival"
            cfg['ANCHOR_SCALES'] = [4, 8, 16, 32]
            cfg['ANCHOR_RATIOS'] = [0.5, 1, 2]
            cfg['MAX_NUM_GT_BOXES'] = 50

        cfg['DEDUP_BOXES'] = cfg['DEDUP_BOXES_numerator'] / cfg['DEDUP_BOXES_denominator']
        cfg['PIXEL_MEANS'] = np.array(cfg['PIXEL_MEANS'])

        if torch.cuda.is_available() and not cfg['CUDA']:
            logger.warning("You have a CUDA device, so you should probably run with --cuda")

        cfg['TRAIN']['USE_FLIPPED'] = True  # Note: Use validation set and disable the flipped to enable faster loading.
        cfg['USE_GPU_NMS'] = cfg['CUDA']
        cfg['EPS'] = float(cfg['EPS'])

        def create_output_path():
            """Return the directory where experimental artifacts are placed.
            If the directory does not exist, it is created.
            """
            project_name = os.path.basename(cfg['ROOT_DIR'])
            if project_name is None or project_name == "":
                project_name = os.path.basename(cfg['ROOT_DIR'])

            outdir = os.path.abspath(os.path.join(os.path.abspath(cfg['OUTPUT_DIR']),
                                                  "{}_{}_{}".format(project_name, cfg['net'], cfg['dataset'])))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            return outdir
        cfg['OUTPUT_PATH'] = create_output_path()

        cfg = edict(cfg)

        logger.info('Called with args:')
        logger.info(pprint.pformat(cfg))

        self._cfg = cfg

    def __str__(self):
        return pprint.pformat(self._cfg)

    @property
    def output_path(self):
        return self._cfg['OUTPUT_PATH']

    def get_ckpt_path(self, epoch_num=None):
        if epoch_num:
            epoch = self._cfg['checkepoch']
        else:
            epoch = epoch_num
        file_name = 'faster_rcnn_{}_{}_{}.pth'.format(self._cfg['checksession'], epoch, self._cfg['checkpoint'])
        ckpt_path = os.path.join(self._cfg['OUTPUT_PATH'], file_name)
        return ckpt_path

    def __getitem__(self, key):
        return self._cfg[key]

    def __getattr__(self, attr):
        # note: only called what self.attr doesn't exist
        try:
            return self._cfg[attr]
        except AttributeError:
            raise Exception("{} does not exist in Config.".format(attr))


cfg = ConfigProvider()
