from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pprint
from time import gmtime, strftime

import numpy as np
import os
import torch
import yaml
from easydict import EasyDict as edict

logger = logging.getLogger(__name__)


class ConfigProvider(dict):
    def __init__(self):
        self._cfg = edict({})
        self._cfg['start_run_time_str'] = strftime("%Y_%b_%d_%H_%M", gmtime())

    def load(self, config_path):
        """Load a config file and override defaults."""
        cfg = {}
        with open('defaults.yml', 'r') as f:
            cfg = yaml.load(f)

        if config_path:
            with open(os.path.expanduser(config_path), 'r') as f:
                model_cfg = yaml.load(f)

            if model_cfg:
                for k, v in model_cfg.items():
                    if k == 'TRAIN' or k == 'TEST':  # TODO can ask if len > 1
                        for k1, v1 in v.items():  # TODO note that this is not recursion...
                            cfg[k][k1] = v1
                    else:
                        cfg[k] = v

        cfg['ROOT_DIR'] = os.path.expanduser(os.path.join(os.path.dirname(__file__), '..'))  # todo check

        cfg['DATA_DIR'] = os.path.expanduser(cfg['DATA_DIR'])

        def create_output_path():
            """Return the directory where experimental artifacts are placed.
            If the directory does not exist, it is created.
            """
            outdir = os.path.join(os.path.expanduser(cfg['OUTPUT_DIR']), cfg['EXPERIMENT_NAME'])
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            return outdir
        cfg['OUTPUT_PATH'] = create_output_path()

        cfg['imdb_name'] = cfg['dataset']['imdb_name']
        cfg['imdbval_name'] = cfg['dataset']['imdbval_name']
        cfg['ANCHOR_SCALES'] = cfg['dataset']['ANCHOR_SCALES']
        cfg['ANCHOR_RATIOS'] = cfg['dataset']['ANCHOR_RATIOS']
        cfg['MAX_NUM_GT_BOXES'] = cfg['dataset']['MAX_NUM_GT_BOXES']

        cfg['DEDUP_BOXES'] = float(cfg['DEDUP_BOXES_numerator']) / float(cfg['DEDUP_BOXES_denominator'])

        cfg['PIXEL_MEANS'] = np.array(cfg['PIXEL_MEANS'])

        cfg['EPS'] = float(cfg['EPS'])

        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

        if torch.cuda.is_available() and not cfg['CUDA']:  # todo del
            logger.warning("You have a CUDA device, so you should probably run with --cuda")

        def create_from_dict(cfg_dict):
            cfg = edict(cfg_dict)
            logger.info('Called with args:')
            logger.info(pprint.pformat(cfg))
            with open('run_with_config.yml', 'w') as outfile:
                yaml.dump(cfg, outfile, default_flow_style=False)

            return cfg
        self._cfg = create_from_dict(cfg)

    def __str__(self):
        return pprint.pformat(self._cfg)

    @property
    def output_path(self):
        return self.OUTPUT_PATH

    def get_ckpt_path(self, epoch_num=None):
        if epoch_num is None:
            epoch = self.checkepoch
        else:
            epoch = epoch_num
        file_name = self.ckpt_file_format.format(epoch)
        path = os.path.join(self.output_path, file_name)
        logger.info("get_ckpt_path: {}.".format(path))
        return path

    def get_last_ckpt_path(self):
        import glob
        list_of_files = glob.glob(os.path.join(self.output_path, self.ckpt_file_format('*')))
        # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    def get_preds_path(self, epoch_num):
        file_name = self.raw_preds_file_format.format(epoch_num)
        path = os.path.join(self.output_path, file_name)
        return path

    def get_postprocessed_detections_path(self, epoch_num):
        file_name = self.postprocessed_file_format.format(epoch_num)
        path = os.path.join(self.output_path, file_name)
        return path

    def get_evals_dir_path(self, epoch_num):
        dir_name = self.evals_dir_format.format(epoch_num)
        # TODO should create dir?
        dir_path = os.path.join(self.output_path, dir_name)
        return dir_path

    def get_img_visualization_path(self, epoch_num, im_num):
        rel_file_path = self.vis_path_format.format(epoch_num, im_num)
        # TODO should create dir?
        full_file_path = os.path.join(self.output_path, rel_file_path)
        return full_file_path

    def get_log_path(self):
        file_name = '{}.log'.format(self.start_run_time_str)
        path = os.path.join(self.output_path, file_name)
        return path

    def __getitem__(self, key):
        return self._cfg[key]

    def __getattr__(self, attr):
        # note: this is called what self.attr doesn't exist
        try:
            return self._cfg[attr]
        except AttributeError:
            raise Exception("{} does not exist in Config.".format(attr))

    def get_state_dict(self):
        return self._cfg
