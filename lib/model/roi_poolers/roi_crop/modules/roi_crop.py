from torch.nn.modules.module import Module

from ..functions.roi_crop import RoICropFunction


class _RoICrop(Module):
    def __init__(self, pool_size, crop_resize_with_max_pool, layout='BHWD'):
        super(_RoICrop, self).__init__()
        self.grid_size = pool_size * 2 if crop_resize_with_max_pool else pool_size

    def forward(self, input1, input2):
        return RoICropFunction()(input1, input2)
