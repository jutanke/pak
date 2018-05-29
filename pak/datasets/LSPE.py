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
#  LSPE
# =========================================
class LSPE(Dataset):
    """ Leeds Sports Pose Extended
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, 'lspe', root, verbose)
        url = 'http://sam.johnson.io/research/lspet_dataset.zip'
        self.root_export = join(root, "lspe")
        self.download_and_unzip(url)

    def get_raw(self):
        """ gets the raw data without any resizing.
            Attention: Image sizes vary!
        """
        image_folder = join(self.root_export, 'images')
        joint_mat = join(self.root_export, 'joints.mat')

        Y = np.rollaxis(loadmat(joint_mat)['joints'], 2, 0)
        n, _, _ = Y.shape

        imgs = sorted([f for f in listdir(image_folder) if \
                        isfile(join(image_folder, f))])
        X = [imread(join(image_folder, f)) for f in imgs]

        return X, Y
