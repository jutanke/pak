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


class DukeMTMC_reID(Dataset):
    """ DukeMTMC-reID dataset, very similar to Market1501
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, "DukeMTMC-reID", root, verbose)
        Dataset.unzip(self)
        self.root_export = join(root, "DukeMTMC-reID")

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
        X = np.array([imread(join(loc, f)) for f in imgs])

        identities = np.array([int(f[0:4]) for f in imgs])
        cameras = np.array([int(f[6]) for f in imgs])
        frames = np.array([int(f[9:15]) for f in imgs])

        Y = np.vstack((identities, cameras, frames))

        return X, np.rollaxis(Y, 1)
