import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F


class AffineGridGenFunction(Function):
    def __init__(self, height, width, lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = input1
        output = input1.new(torch.Size([input1.size(0)]) + self.grid.size()).zero_()
        self.batchgrid = input1.new(torch.Size([input1.size(0)]) + self.grid.size()).zero_()
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid.astype(self.batchgrid[i])

        for i in range(input1.size(0)):
            output = torch.bmm(self.batchgrid.view(-1, self.height * self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output

    def backward(self, grad_output):
        grad_input1 = self.input1.new(self.input1.size()).zero_()
        grad_input1 = torch.baddbmm(grad_input1,
                                    torch.transpose(grad_output.view(-1, self.height*self.width, 2), 1, 2),
                                    self.batchgrid.view(-1, self.height * self.width, 3))
        return grad_input1


def _affine_grid_gen(rois, input_size, grid_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([(x2 - x1) / (width - 1),
                       zero,
                       (x1 + x2 - width + 1) / (width - 1),
                       zero,
                       (y2 - y1) / (height - 1),
                       (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))
    return grid


