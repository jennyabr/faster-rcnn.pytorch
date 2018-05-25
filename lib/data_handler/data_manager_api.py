from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch.autograd import Variable


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class DataManager(ABC):
    def __init__(self, mode, is_cuda):
        self.is_test = mode == Mode.TEST
        self.is_train = mode == Mode.TRAIN

        def create_input_tensors():
            im_data = torch.FloatTensor(1)
            im_info = torch.FloatTensor(1)
            num_boxes = torch.LongTensor(1)
            gt_boxes = torch.FloatTensor(1)

            if is_cuda:
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                num_boxes = num_boxes.cuda()
                gt_boxes = gt_boxes.cuda()

            im_data = Variable(im_data, volatile=self.is_test)
            im_info = Variable(im_info, volatile=self.is_test)
            num_boxes = Variable(num_boxes, volatile=self.is_test)
            gt_boxes = Variable(gt_boxes, volatile=self.is_test)

            return im_data, im_info, gt_boxes, num_boxes
        self.im_data, self.im_info, self.gt_boxes, self.num_boxes = create_input_tensors()

        # TODO JA what to do here?
        self.data_iter = None

    @abstractmethod
    def transform_data_tensors(self, data):  # TODO maybe (self, im_data, im_info, gt_boxes, num_boxes) the order depends on how the data is constracted...
        #return self.im_data, self.im_info, self.gt_boxes, self.num_boxes
        raise NotImplementedError

    def __next__(self):
        data = next(self.data_iter)
        self.im_data, self.im_info, self.gt_boxes, self.num_boxes = self.transform_data_tensors(data)
        return self.im_data, self.im_info, self.gt_boxes, self.num_boxes

    def __len__(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError