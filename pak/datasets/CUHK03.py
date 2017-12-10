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


class cuhk03(Dataset):
    """ cuhk03 dataset
    """

    def __init__(self, root, verbose=True, target_w=100, target_h=256):
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
        self.target_w = target_w
        self.target_h = target_h


    def get_detected(self):
        """ gets the images that were detected by a automated person detector
        """
        return self.get_raw('detected')

    def get_labeled(self):
        """ gets the images that were humanly-annotated
        """
        return self.get_raw('labeled')

    def get_raw(self, folder):
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
                        img = np.swapaxes(img, 0, 2)
                        img = resize(img, (th, tw), mode='constant')
                        Imgs.append((img * 255).astype('uint8'))
                        Ids.append(current_id)
                current_id += 1

        X = np.array(Imgs)
        Y = np.array(Ids, 'int32')
        return X, Y
