from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch.autograd import Variable


class Mode(Enum):
    TRAIN = 1
    INFER = 2


class DataManager(ABC):
    def __init__(self, mode, is_cuda):
        self.is_infer = mode == Mode.INFER
        self.is_train = mode == Mode.TRAIN

        def create_input_tensors():
            im_data = torch.FloatTensor(1)  # TODO JA check the param [1] - is it needed?...
            im_info = torch.FloatTensor(1)
            num_boxes = torch.LongTensor(1)
            gt_boxes = torch.FloatTensor(1)

            if is_cuda:
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                num_boxes = num_boxes.cuda()
                gt_boxes = gt_boxes.cuda()

            im_data = Variable(im_data, volatile=self.is_infer)
            im_info = Variable(im_info, volatile=self.is_infer)
            num_boxes = Variable(num_boxes, volatile=self.is_infer)
            gt_boxes = Variable(gt_boxes, volatile=self.is_infer)

            return im_data, im_info, gt_boxes, num_boxes
        self._im_data, self._im_info, self._gt_boxes, self._num_boxes = create_input_tensors()

        # TODO JA what to do here?
        self._data_iter = None

    @abstractmethod
    def transform_data_tensors(self, data):
        # TODO JA maybe (self, im_data, im_info, gt_boxes, num_boxes) the order
        # TODO JA depends on how the data is constracted...
        # return self.im_data, self.im_info, self.gt_boxes, self.num_boxes
        raise NotImplementedError

    def __next__(self):
        data = next(self._data_iter)
        self._im_data, self._im_info, self._gt_boxes, self._num_boxes = self.transform_data_tensors(data)
        return self._im_data, self._im_info, self._gt_boxes, self._num_boxes

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def classes(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def data_loader(self):
        raise NotImplementedError

    def prepare_iter_for_new_epoch(self):
        self._data_iter = iter(self.data_loader)
