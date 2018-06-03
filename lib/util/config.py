from __future__ import absolute_import
from __future__ import division

import logging
import pprint
from time import gmtime, strftime

import numpy as np
import os
import random
import torch
import yaml
from easydict import EasyDict as edict

logger = logging.getLogger(__name__)

# TODO: ConfigProvider should enable setting attributes inside _cfg for h.p. sweeps
class ConfigProvider(dict):
    def __init__(self):
        super(ConfigProvider, self).__init__()
        self._cfg = edict({})

    def load(self, config_path):
        """Load a config file and override defaults."""
        with open(os.path.join(os.path.dirname(__file__), 'defaults.yml'), 'r') as f:
            cfg = yaml.load(f)

        if config_path:
            with open(os.path.abspath(config_path), 'r') as f:
                model_cfg = yaml.load(f)

            if model_cfg:
                for k, v in model_cfg.items():
                    if k == 'TRAIN' or k == 'TEST':  # TODO can ask if len > 1
                        for k1, v1 in v.items():  # TODO note that this is not recursion...
                            cfg[k][k1] = v1
                    else:
                        cfg[k] = v

        dataset = cfg['dataset']
        cfg['imdb_name'] = cfg[dataset]['imdb_name']
        cfg['imdbval_name'] = cfg[dataset]['imdbval_name']
        cfg['ANCHOR_SCALES'] = cfg[dataset]['ANCHOR_SCALES']
        cfg['ANCHOR_RATIOS'] = cfg[dataset]['ANCHOR_RATIOS']
        cfg['MAX_NUM_GT_BOXES'] = cfg[dataset]['MAX_NUM_GT_BOXES']

        cfg['DEDUP_BOXES'] = float(cfg['DEDUP_BOXES_numerator']) / float(cfg['DEDUP_BOXES_denominator'])

        self.create_from_dict(cfg)

    def create_from_dict(self, cfg):
        cfg['start_run_time_str'] = strftime("%Y_%b_%d_%H_%M", gmtime())

        cfg['PIXEL_MEANS'] = np.array(cfg['PIXEL_MEANS'])

        cfg['EPS'] = float(cfg['EPS'])

        cfg['DATA_DIR'] = os.path.abspath(cfg['DATA_DIR'])

        def create_output_path():
            """Return the directory where experimental artifacts are placed.
            If the directory does not exist, it is created.
            """
            outdir = os.path.join(os.path.abspath(cfg['OUTPUT_DIR']), cfg['EXPERIMENT_NAME'])
            os.makedirs(outdir, exist_ok=True)
            return outdir
        cfg['OUTPUT_PATH'] = create_output_path()
        seed = cfg.get('RNG_SEED', random.randint(1,1e20)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        cfg = edict(cfg)
        logger.info('--->>> Config:\n{}'.format(pprint.pformat(cfg)))
        with open(os.path.join(cfg['OUTPUT_PATH'], 'run_with_config.yml'), 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)

        self._cfg = cfg

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info("CKPT path: {}.".format(path))
        return path

    def get_last_ckpt_path(self):
        import glob
        list_of_files = glob.glob(os.path.join(self.output_path, self.ckpt_file_format.format('*')))
        latest_file = max(list_of_files, key=os.path.getctime)
        
        return latest_file

    def get_preds_path(self, epoch_num):
        file_name = self.raw_preds_file_format.format(epoch_num)
        path = os.path.join(self.output_path, file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_postprocessed_detections_path(self, epoch_num):
        file_name = self.postprocessed_file_format.format(epoch_num)
        path = os.path.join(self.output_path, file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_evals_dir_path(self, epoch_num):
        dir_name = self.evals_dir_format.format(epoch_num)
        dir_path = os.path.join(self.output_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def get_img_visualization_path(self, epoch_num, im_num):
        rel_file_path = self.vis_path_format.format(epoch_num, im_num)
        full_file_path = os.path.join(self.output_path, rel_file_path)
        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
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
