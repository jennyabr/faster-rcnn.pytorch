import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from data_handler.data_manager_api import DataManager, Mode
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb


class BDSampler(Sampler):
    def __init__(self, train_size, batch_size, seed):
        super(BDSampler, self).__init__(data_source="")  # TODO what to do with data_source?
        self.seed = seed
        torch.manual_seed(seed)

        self.data_size = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.data_size  # TODO (self.data_size + self.batch_size - 1) // self.batch_size


class FasterRCNNDataManager(DataManager):
    def __init__(self, mode, imdb_name, seed, num_workers, is_cuda, cfg, batch_size=1):
        super(FasterRCNNDataManager, self).__init__(mode, is_cuda)
        self._imdb, roidb, ratio_list, ratio_index = combined_roidb(
            imdb_name,
            use_flipped=cfg.TRAIN.USE_FLIPPED,
            proposal_method=cfg.TRAIN.PROPOSAL_METHOD,
            training=self.is_train,
            data_dir=cfg.DATA_DIR)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size,
                                 self.imdb.num_classes, cfg, training=self.is_train)
        self.batch_size = batch_size

        if self.is_train:
            self._train_size = train_size = len(roidb)
            sampler_batch = BDSampler(train_size, batch_size, seed)
            self._data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                           sampler=sampler_batch)
            self.iters_per_epoch = int(self._train_size / batch_size)
        elif mode == Mode.INFER:
            self.imdb.competition_mode(on=True)  # TODO this function is not implemented...
            self._data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                           shuffle=False, pin_memory=True)
        else:
            raise Exception("Not valid mode {} - should be TRAIN or TEST".format(mode))

    def transform_data_tensors(self, data):
        self._im_data.data.resize_(data[0].size()).copy_(data[0])
        self._im_info.data.resize_(data[1].size()).copy_(data[1])
        self._gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        self._num_boxes.data.resize_(data[3].size()).copy_(data[3])
        return self._im_data, self._im_info, self._gt_boxes, self._num_boxes

    def __len__(self):
        return len(self._imdb.image_index)

    @property
    def num_classes(self):
        return self._imdb.num_classes

    @property
    def classes(self):
        return self._imdb.classes

    @property
    def data_loader(self):
        return self._data_loader

    @property
    def num_images(self):
        return self._imdb.num_images

    @property
    def imdb(self):
        return self._imdb

