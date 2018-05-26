from lib.model.roi_align.modules.roi_align import RoIAlignAvg
from lib.model.roi_crop.modules.roi_crop import _RoICrop
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from cfgs.config import cfg #TODO: this class shouldn't import cfg

#TODO put all rio in one dir
def create_roi_pooler(roi_pooler_name):
    if roi_pooler_name== 'crop':
        roi_pooler = _RoICrop()
    elif roi_pooler_name == 'align':
        roi_pooler = RoIAlignAvg(cfg.roi_pooler_size, cfg.roi_pooler_size, 1.0/16.0)
    elif roi_pooler_name == 'pool':
        roi_pooler = _RoIPooling(cfg.roi_pooler_size, cfg.roi_pooler_size, 1.0/16.0)
    return roi_pooler
