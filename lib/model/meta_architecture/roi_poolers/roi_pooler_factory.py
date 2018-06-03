from model.meta_architecture.roi_poolers.roi_align.modules.roi_align import RoIAlignAvg
from model.meta_architecture.roi_poolers.roi_crop.modules.roi_crop import _RoICrop
from model.meta_architecture.roi_poolers.roi_pooling.modules.roi_pool import _RoIPooling


def create_roi_pooler(roi_pooler_name, roi_pooler_size, crop_resize_with_max_pool=None):
    if roi_pooler_name == 'crop':
        roi_pooler = _RoICrop(roi_pooler_size, crop_resize_with_max_pool)
    elif roi_pooler_name == 'align':
        roi_pooler = RoIAlignAvg(roi_pooler_size, roi_pooler_size, 1.0/16.0)
    elif roi_pooler_name == 'pool':
        roi_pooler = _RoIPooling(roi_pooler_size, roi_pooler_size, 1.0/16.0)
    else:
        raise Exception("Unexpected rio model {} - should be one of [crop, align, pool]".format(roi_pooler_name))
    return roi_pooler
