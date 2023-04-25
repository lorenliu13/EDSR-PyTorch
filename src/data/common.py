import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    """
    *args: a tuple of input images, the first images is used to determine the height and width for patch extraction
    path_size: the size of low-resolution patch, if input_large is ture
    scale: scale between low and high resolution images

    input_large: whether the input image is already in high-resoluiton format, usually it is false
    Randomly extracts a patch from one or more input images.
    The patch size will be smaller than the original image.
    """
    ih, iw = args[0].shape[:2] # get the dimension of the first input image

    if not input_large:
        # determine the size of the high-resolution images (tp)
        # determine the size of the low-resolution images (ip)
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    # calculate the top-left corner coordinates (ix, iy) for the low-resolution patch
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    # calculate the top-left corner coordinates (tx, ty) for the high-resoluiton patch
    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    # extract patches from the input images and store them in the ret list.
    ret = [
        args[0][iy:iy + ip, ix:ix + ip], # extract a patch from the first lr image with size ip
        *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]] # extract patches from hr images with size tp
    ]
    # This is for two dimensional array

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2: # if the image is 2-dimension
            img = np.expand_dims(img, axis=2) # add a new dimension

        c = img.shape[2] # stores the number of color channels
        if n_channels == 1 and c == 3: # if n_channel = 1 but c = 3
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2) # convert RGB to YCbCr colorspace
        elif n_channels == 3 and c == 1: # if n_channel = 3 but c = 1
            img = np.concatenate([img] * n_channels, 2) # replicate the image three times

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


def np2Tensor_npy(*args, rgb_range=1):
    """
    Convert numpy file to tensor array and rescale it by the rgb_range factor.
    """
    def _np2Tensor_npy(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range)

        return tensor

    return [_np2Tensor_npy(a) for a in args]





def augment(*args, hflip=True, rot=True):
    """
    Apply three types of data augmentation to the input images:
    Horizontal flip
    Vertical flip
    90 degree rotation

    """
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        """
        This is only for 2-d images
        """
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        if rot90: img = img.transpose(1, 0)
        
        return img

    return [_augment(a) for a in args]

