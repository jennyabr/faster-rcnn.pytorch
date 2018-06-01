import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module
import torch.nn.functional as F

from model.utils.net_utils import _affine_grid_gen
from ..functions.roi_crop import RoICropFunction


class _RoICrop(Module):
    def __init__(self, pool_size, crop_resize_with_max_pool, layout='BHWD'):
        super(_RoICrop, self).__init__()
        self.grid_size = pool_size * 2 if crop_resize_with_max_pool else pool_size

    def forward(self, features, rois):
        grid_xy = _affine_grid_gen(rois, features.size()[2:], self.grid_size)
        grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
        pooled_rois = RoICropFunction()(features, Variable(grid_yx).detach())
        if self.cfg_params['crop_resize_with_max_pool']:
            pooled_rois = F.max_pool2d(pooled_rois, 2, 2)
        return pooled_rois
