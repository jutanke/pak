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

# =========================================
#  MARKET 1501
# =========================================

class Market1501(Dataset):
    """ Market1501 dataset
    """

    def __init__(self, root, verbose=True):
        url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
        Dataset.__init__(self, "Market-1501-v15.09.15", root, verbose)
        self.download_and_unzip(url)
        self.root_export = join(root, "Market-1501-v15.09.15")

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
        X = np.array([imread(join(loc, f)) for f in imgs], 'uint8')

        identities = np.array([int(f[0:2]) if f.startswith('-1') else int(f[0:4]) \
            for f in imgs])
        cameras = np.array([int(f[4]) if f.startswith('-1') else int(f[6]) \
            for f in imgs])
        sequences = np.array([int(f[6]) if f.startswith('-1') else int(f[8]) \
            for f in imgs])
        frames = np.array([int(f[8:14]) if f.startswith('-1') else int(f[10:16]) \
            for f in imgs])

        Y = np.vstack((identities, cameras, sequences, frames))

        return X, np.rollaxis(Y, 1)
