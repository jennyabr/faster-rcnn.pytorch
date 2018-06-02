def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
             num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
             timeout=0, worker_init_fn=None):
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.collate_fn = collate_fn
    self.pin_memory = pin_memory
    self.drop_last = drop_last
    self.timeout = timeout
    self.worker_init_fn = worker_init_fn

    if timeout < 0:
        raise ValueError('timeout option should be non-negative')

    if batch_sampler is not None:
        if batch_size > 1 or shuffle or sampler is not None or drop_last:
            raise ValueError('batch_sampler is mutually exclusive with '
                             'batch_size, shuffle, sampler, and drop_last')

    if sampler is not None and shuffle:
        raise ValueError('sampler is mutually exclusive with shuffle')

    if self.num_workers < 0:
        raise ValueError('num_workers cannot be negative; '
                         'use num_workers=0 to disable multiprocessing.')

    if batch_sampler is None:
        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    self.sampler = sampler
    self.batch_sampler = batch_sampler


def __iter__(self):
    return DataLoaderIter(self)


def __len__(self):
    return len(self.batch_sampler)
