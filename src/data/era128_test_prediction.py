import os

from data import common
from data import srdata_era

import numpy as np

import torch
import torch.utils.data as data
import glob


class era128_test_prediction(srdata_era.SRData):
    """
    Define a class to load .npy data for 32 to 128 task
    """

    def __init__(self, args, name='', train=True, benchmark=True):
        super(era128_test_prediction, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        matching_directories = glob.glob(
            os.path.join(dir_data, '*era5_32_128_test'))  # * is the wild card for ensemble year and id
        # take the first match
        self.apath = matching_directories[0]
        # self.apath = os.path.join(dir_data, 'era5_32_128_test')
        self.dir_hr = os.path.join(self.apath, 'lr_32')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'lr_32')
        else:
            self.dir_lr = os.path.join(self.apath, 'lr_32')
        self.ext = ('', '.npy')

