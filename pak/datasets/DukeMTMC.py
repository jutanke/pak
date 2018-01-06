from pak.datasets.Dataset import Dataset
import numpy as np
import zipfile
import tarfile
import urllib.request
import shutil
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.io import loadmat
from skimage.transform import resize
from pak import utils
from pak.util import mpii_human_pose as mpii_hp
import h5py
from enum import Enum
import time


class DukeMTMC_reID(Dataset):
    """ DukeMTMC-reID dataset, very similar to Market1501
    """

    def __init__(self, root, verbose=True, force_shape=None, memmapped=True):
        Dataset.__init__(self, "DukeMTMC-reID", root, verbose)
        Dataset.unzip(self)
        self.root_export = join(root, "DukeMTMC-reID")
        self.memmapped = memmapped

        if force_shape is None:
            self.force_shape = None
        else:
            assert len(force_shape) == 2
            w,h = force_shape
            self.force_shape = (h, w)  # as resize used inversed logic..

    def can_be_memmapped(self):
        """ as memmorymapped files need a strict structure
            we cannot just store unresized images as files
        """
        return self.force_shape is not None

    def get_memmapped_file_name(self, folder):
        assert self.can_be_memmapped()
        force_shape = self.force_shape
        shape_str = str(force_shape[0]) + 'x' + str(force_shape[1])
        file_name = folder + shape_str  + '.npy'
        return join(self.root_export , file_name)

    def get_y_file(self, folder):
        file_name = folder + "_y.npy"
        return join(self.root_export, file_name)

    def get_memmapped_file_shape(self, folder):
        assert self.can_be_memmapped()
        if folder == 'bounding_box_train':
            n = 16522
        elif folder == 'bounding_box_test':
            n = 17661
        else:
            raise Exception("Cannot find subsystem " + folder)

        h,w = self.force_shape
        return (n, h, w, 3)

    def get_train(self):
        return self.get_raw('bounding_box_train')

    def get_test(self):
        return self.get_raw('bounding_box_test')

    def get_raw(self, folder):
        """ gets the raw identity pictures
        """
        loc = join(self.root_export, folder)
        imgs = sorted([f \
                    for f in listdir(loc) if isfile(join(loc, f)) and \
                      f.endswith('jpg')])

        X_is_loaded = False
        is_memmapped = self.can_be_memmapped() and self.memmapped
        if is_memmapped:
            fmmap = self.get_memmapped_file_name(folder)
            data_shape = self.get_memmapped_file_shape(folder)
            if isfile(fmmap):
                utils.talk('load memmap ' + fmmap, self.verbose)
                X = np.memmap(fmmap, dtype='uint8', mode="r", shape=data_shape)
                X_is_loaded = True

        Y_is_loaded = False
        fydata = self.get_y_file(folder)
        if isfile(fydata):
            Y = np.load(fydata)
            Y_is_loaded = True

        if X_is_loaded and Y_is_loaded:
            return X, Y

        if not X_is_loaded:
            force_shape = self.force_shape

            if force_shape is None:
                X = np.array([imread(join(loc, f)) for f in imgs])
            else:
                X = np.array([imresize(imread(join(loc, f)), size=force_shape) \
                    for f in imgs])

            if is_memmapped:
                # store X to file
                utils.talk('create memmap ' + fmmap, self.verbose)
                X_ = np.memmap(fmmap, dtype='uint8', mode="w+", shape=data_shape)
                for i, x in enumerate(X):
                    X_[i] = x

                utils.talk('flush memmaped ' + fmmap, self.verbose)
                del X_  # flush
                time.sleep(3)

                X = np.memmap(fmmap, dtype='uint8', mode="r", shape=data_shape)

        if not Y_is_loaded:
            identities = np.array([int(f[0:4]) for f in imgs])
            cameras = np.array([int(f[6]) for f in imgs])
            frames = np.array([int(f[9:15]) for f in imgs])

            Y = np.vstack((identities, cameras, frames))
            Y = np.rollaxis(Y, 1)

            np.save(fydata, Y)

        return X, Y
