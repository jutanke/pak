from pak.datasets.Dataset import Dataset
import numpy as np
import zipfile
import tarfile
import urllib.request
import shutil
import pickle as pkl
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


class MPII_human_pose(Dataset):
    """ MPII Human Pose dataset
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, 'mpii_human_pose_v1', root, verbose)

        url_data = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz'
        url_anno = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip'
        self.download_and_unzip(url_anno,
            zipfile_name='mpii_human_pose_v1_u12_2.zip',
            dest_folder='mpii_human_pose_v1_u12_2')
        self.root_export = join(root, 'mpii_human_pose_v1_u12_2')
        self.download_and_unzip(url_data,
            zipfile_name='mpii_human_pose_v1.tar.gz',
            dest_folder=join('mpii_human_pose_v1_u12_2', 'images'))

        # -- load images --
        mpii_root = self.root_export

        img_dir = join(mpii_root, 'images/images')
        assert isdir(img_dir)

        n = len(listdir(img_dir))

        fmmap = join(mpii_root, 'X.npy')
        flookup = join(mpii_root, 'lookup.pkl')

        lookup = []
        if not isfile(fmmap):
            assert not isfile(flookup)
            X = np.memmap(fmmap, dtype='uint8', mode='w+',
                          shape=(n, 1080, 1920, 3))
            for idx, f in enumerate(sorted(listdir(img_dir))):
                if idx % 100 == 0:
                    print(str(idx) + '/' + str(n))

                I = imread(join(img_dir, f))
                h,w,_ = I.shape
                X[idx,0:h,0:w,:] = I
                lookup.append((f,h,w))

            with open(flookup, 'wb') as f:
                pkl.dump(lookup, f)

            del X  # flush data

        self.X = np.memmap(fmmap, dtype='uint8', mode='r',
                      shape=(n, 1080, 1920, 3))

        with open(flookup, 'rb') as f:
            self.lookup = pkl.load(f)


    def get_annotation(self):
        """ reads the annotation and returns it
        """
        annot_dir = join(self.root_export, 'mpii_human_pose_v1_u12_2')
        mat = join(annot_dir, "mpii_human_pose_v1_u12_1.mat")
        M = loadmat(mat)
        M = M['RELEASE']
        AL = M['annolist'][0][0][0]
        TR = M['img_train'][0][0][0]


        n = len(AL)
        n = 10
        result = []
        for i in range(n):
            print("stuff", i)
            e = AL[i]   # get the image meta data, nbr of persons,
                        # person joints, etc.
            print(e)
            is_training_data = TR[i] == 1
            data = mpii_hp.get_data(e, is_training_data)
            result.append(data)
        return result
