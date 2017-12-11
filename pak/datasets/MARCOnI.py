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
import h5py
from enum import Enum

class MARCOnI(Dataset):
    """ MPII http://marconi.mpi-inf.mpg.de/#download
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, 'marconi', root, verbose)
        self.root_export = join(root, 'marconi')
        self.download_set("Soccer")
        self.download_set("Kickbox")
        self.download_set("SBoard")
        self.download_set("Soccer2")
        self.download_set("Walk1")
        self.download_set("Walk2")
        self.download_set("Volleyball")
        self.download_set("Juggling")
        self.download_set("Run2")

    def download_set(self, name):
        """ download the given set
        """
        self.root_export = join(self.root, 'marconi/' + name)
        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + '/Images.zip'
        self.download_and_unzip(url, zipfile_name='Images.zip',
            dest_folder='marconi/' + name + "/Images", dest_force=True,
            root_folder='marconi/' + name)

        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + '/CNN_Detections.zip'
        self.download_and_unzip(url, zipfile_name='CNN_Detections.zip',
            dest_folder='marconi/' + name + "/CNN_Detections", dest_force=True,
            root_folder='marconi/' + name)

        self.root_export = join(self.root, 'marconi')
        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + \
            '/Annotations.mat'
        self.download_file(url, file_name='Annotations.mat', dest_folder=name)

        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + \
            '/Calibration.dat'
        self.download_file(url, file_name='Calibration.dat', dest_folder=name)
