# Define a new function for reading npy data
import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name # the name of the dataset
        self.train = train # boolean indicating the dataset is for training or testing, default is True
        self.split = 'train' if train else 'test' # the current phase
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR') # it is true if model dataset is VDSR
        self.scale = args.scale
        self.idx_scale = 0
        
        self._set_filesystem(args.dir_data) # set up the directory structure for the dataset
        if args.ext.find('img') < 0: # check whether args.ext contains "img", if not, creates a "bin" directory under "self.path"
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan() # call scan method to get the list of HR and LR images
        if args.ext.find('img') >= 0 or benchmark: # if args.ext contains 'img' or benchmark is true, the HR and LR images are stored in self.image_hr and self.image_lr
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0: # otherwise, the images are converted to pytorch tensors and saved as ".pt" files
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True)

        # if train is true, calculate the number of times each HR image need to the repeated in an epoch
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr) # total number of training images
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        # search all the files in the dir_hr directory that have the extension self.ext[0]
        # sorted function is then applied to the list of file names to sort them alphabetically
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        # empty list for lr
        names_lr = [[] for _ in self.scale]
        # the LR image file name is by joining the dir_lr and scaling factor subdirectory
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f)) # get the file name without extension
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{0}{1}'.format(
                        filename, self.ext[1]
                    ) # get the absolute file path of lr images
                ))

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        """
        converting an image file to a binary file and save it
        """
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            # if the binary file not exist
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        """
        Retrieves an image pair at the index 'idx'
        """
        lr, hr, filename = self._load_file(idx) # load the lr and hr images
        pair = self.get_patch(lr, hr) # extract patches from the LR and HR images
        pair = common.set_channel(*pair, n_channels=self.args.n_colors) # modify the channel number
        pair_t = common.np2Tensor_npy(*pair, rgb_range=1) # convert from numpy array to tensor.
        # The numpy array is already between 0 and 1, there is no need to rescale it

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr)) # get the filename without extension
        if self.args.ext == 'npy' or self.benchmark: #  If self.args.ext is 'img' or self.benchmark is True
            hr = np.load(f_hr) # load data from numpy objects
            lr = np.load(f_lr) # load data from numpy objects
        elif self.args.ext.find('sep') >= 0: #  If self.args.ext contains 'sep', it implies that the image data is stored in a separate file format
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        '''
        Get small patches from training images
        '''

        scale = self.scale[self.idx_scale]
        if self.train: # if the phase is training, use a patch
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large # usually it is false
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr) # if no_augment is false, apply data augmentation
        else: # if in the testing phase, use the original size
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

