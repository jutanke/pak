from pak.datasets.Dataset import Dataset
import numpy as np
import zipfile
import tarfile
import urllib.request
import shutil
import time
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


class cuhk03(Dataset):
    """ cuhk03 dataset
    """

    def __init__(self, root, verbose=True, target_w=100, target_h=256, memmapped=True):
        """ create cuhk03 dataset

            root: data folder that stores all files
            verbose: boolean flag to tell if the class should talk in debug mode
                or not
            target_w: resize all images to fit target_w
            target_h: resize all images to fit target_h
        """
        Dataset.__init__(self, "cuhk03_release", root, verbose)
        Dataset.unzip(self)
        self.hd5file = join(join(root, self.name), 'cuhk-03.mat')
        self.memmapped_file_location = join(root, self.name)
        self.target_w = int(target_w)
        self.target_h = int(target_h)
        self.memmapped = memmapped


    def get_memmapped_file_name(self, folder):
        """ type is either 'detected' or 'labeled'
        """
        file_name = folder + str(self.target_w) + "x" + str(self.target_h) +\
            '.npy'
        return join(self.memmapped_file_location, file_name)


    def get_y_file(self, folder):
        """ gives the filename for the y-data
        """
        file_name = folder + "_y.npy"
        return join(self.memmapped_file_location, file_name)


    def get_memmapped_file_shape(self, folder):
        """ gives the right size
        """
        if folder == 'detected':
            n = 14097
        elif folder == 'labeled':
            n = 14096
        else:
            raise Exception("Could not find shape for folder " + folder)
        return (n, self.target_h, self.target_w, 3)


    def get_detected(self):
        """ gets the images that were detected by a automated person detector
        """
        return self.get_raw('detected')

    def get_labeled(self):
        """ gets the images that were humanly-annotated
        """
        return self.get_raw('labeled')

    def get_raw(self, folder):

        # step 1: check if we already have memmapped file:
        X_is_loaded = False
        if self.memmapped:
            fmmap = self.get_memmapped_file_name(folder)
            if isfile(fmmap):
                utils.talk('found memmaped ' + fmmap, self.verbose)
                data_shape = self.get_memmapped_file_shape(folder)
                X = np.memmap(fmmap, dtype='uint8', mode="r", shape=data_shape)
                X_is_loaded = True
            else:
                utils.talk('could not find memmaped ' + fmmap, self.verbose)

        Y_is_loaded = False
        fydata = self.get_y_file(folder)
        if isfile(fydata):
            Y = np.load(fydata)
            Y_is_loaded = True

        if X_is_loaded and Y_is_loaded:
            return X, Y

        f = h5py.File(self.hd5file,'r+')
        data = f[folder]
        _, n = data.shape
        tw = self.target_w
        th = self.target_h

        current_id = 1

        Imgs = []
        Ids = []

        for view in range(n):
            utils.talk("cuhk03: view \"" + folder + "\" " + \
                str(view+1) + "/" + str(n), self.verbose)
            V = f[data[0][view]]
            ten, M = V.shape  # M is number of identities
            for i in range(M):
                for j in range(ten):
                    img = f[V[j][i]].value
                    if len(img.shape) == 3:
                        if not X_is_loaded:
                            img = np.swapaxes(img, 0, 2)
                            img = resize(img, (th, tw), mode='constant')
                            Imgs.append((img * 255).astype('uint8'))
                        Ids.append(current_id)
                current_id += 1

        if not X_is_loaded:
            X = np.array(Imgs)
            if self.memmapped:


                utils.talk('write memmaped ' + fmmap, self.verbose)
                data_shape = self.get_memmapped_file_shape(folder)
                X_ = np.memmap(fmmap, dtype='uint8', mode="w+", shape=data_shape)
                #X_[:] = X[:]
                for i, x in enumerate(X):
                    X_[i] = x

                utils.talk('flush memmaped ' + fmmap, self.verbose)
                del X_  # flush
                time.sleep(3)

                X = np.memmap(fmmap, dtype='uint8', mode="r", shape=data_shape)

        Y = np.array(Ids, 'int32')

        if not Y_is_loaded:
            np.save(fydata, Y)
        
        return X, Y
