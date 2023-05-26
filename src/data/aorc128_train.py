import os

from data import common
from data import srdata_era

import numpy as np

import torch
import torch.utils.data as data

class aorc128_train(srdata_era.SRData):
    """
    Define a class to load .npy data for 128 to 512 task
    """

    def __init__(self, args, name='', train=True, benchmark=True):
        super(aorc128_train, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'aorc_32_128_train')
        self.dir_hr = os.path.join(self.apath, 'hr_128')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'lr_32')
        else:
            self.dir_lr = os.path.join(self.apath, 'lr_32')
        self.ext = ('', '.npy')

