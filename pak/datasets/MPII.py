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

        sizes_lookup = {}
        
        sizes_lookup_by_fname_pkl = join(mpii_root, 'fname_lookup.pkl')
        sizes_lookup_by_fname = {}
        all_files_are_loaded = isfile(sizes_lookup_by_fname_pkl)
        
        for fname, fshape, imshape in mpii_hp.get_all_shapes():
            fpath = join(mpii_root, fname)
            sizes_lookup_by_fname[imshape] = []
            if not isfile(fpath):
                X = np.memmap(fpath, dtype='uint8', mode='w+', shape=fshape)
                sizes_lookup[imshape] = (X, 0)

        img_dir = join(mpii_root, 'images/images')
        assert isdir(img_dir)
        total = len(listdir(img_dir))
        
        if not all_files_are_loaded:
            for idx, f in enumerate(sorted(listdir(img_dir))):
                if idx % 100 == 0:
                    print(str(idx) + '/' + str(total))
                I = imread(join(img_dir, f))
                shape_str = str(I.shape)
                sizes_lookup_by_fname[shape_str].append(f)
                if shape_str in sizes_lookup:
                    X, i = sizes_lookup[shape_str]
                    X[i] = I
                    sizes_lookup[shape_str] = (X, i+1)
            
            with open(sizes_lookup_by_fname_pkl, 'wb') as f:
                pkl.dump(sizes_lookup_by_fname, f)
        else:
            with open(sizes_lookup_by_fname_pkl, 'rb') as f:
                sizes_lookup_by_fname_pkl = pkl.load(f)
        
        if not all_files_are_loaded:
            for k, val in sizes_lookup.items():
                if val is not None:
                    del val[0]  # flush the data
        
        self.sizes_lookup_by_fname = sizes_lookup_by_fname



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
