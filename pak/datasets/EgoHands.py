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

class EgoHands_config(Enum):
    Polygon = 1 # polygon as in the original data
    AABB = 2    # AABB for simplification


class EgoHands(Dataset):

    def __init__(self, root, verbose=True):
        """ ctro
        """
        Dataset.__init__(self, "egohands_data", root, verbose)
        url = 'http://vision.soic.indiana.edu/egohands_files/egohands_data.zip'
        self.root_export = join(root, "egohands_data")
        self.download_and_unzip(url)


    def get_raw(self, config=EgoHands_config.Polygon, memmapped=False):
        """
        """
        # step 1: get all videos
        labelled_samples_url = join(self.root_export, '_LABELLED_SAMPLES')
        all_videos = [join(labelled_samples_url, f) for f in \
            listdir(labelled_samples_url) if isdir(join(labelled_samples_url, f))]

        # step 2: load video frames and polygon dataset
        Y = []

        if memmapped:
            X_shape = (48, 100, 720, 1280, 3)
            fmmap = join(self.root_export, 'egohands.memmap')
            fmmap_exists = isfile(fmmap)
            if not fmmap_exists:
                X = np.memmap(fmmap, dtype='uint8', mode='w+', shape=X_shape)
        else:
            X = []


        for vindx, vurl in enumerate(all_videos):
            imgs = sorted([f \
                        for f in listdir(vurl) if isfile(join(vurl, f)) and \
                          f.endswith('jpg')])
            assert(len(imgs) == 100)  # sanity check

            if memmapped:
                if not fmmap_exists:
                    # if we already created the memmap file we do NOT
                    # want to recreate it!
                    imgs = np.array([imread(join(vurl, f)) for f in imgs], \
                        'uint8')
                    X[vindx] = imgs
            else:
                imgs = np.array([imread(join(vurl, f)) for f in imgs], 'uint8')
                X.append(imgs)

            polygon_url = join(vurl, 'polygons.mat')
            M = loadmat(polygon_url)['polygons'][0]
            Y_single_video = []
            for i in range(100):
                V = M[i]
                Y_single_frame = []
                for hand in range(4):
                    H = V[hand]
                    #if len(H) > 0:
                    if config is EgoHands_config.Polygon:
                        Y_single_frame.append(H)
                    elif len(H) > 1:  # meaning: hand is not visible
                        x = H[:,0]
                        y = H[:,1]
                        top_right = (np.max(x), np.max(y))
                        bottom_left = (np.min(x), np.min(y))
                        Y_single_frame.append((top_right, bottom_left))
                Y_single_video.append(Y_single_frame)
            Y.append(Y_single_video)

        # step 2: read metadata
        #M = loadmat(join(self.root_export, 'metadata.mat'))
        #
        # M = loadmat(join(self.root_export, '_LABELLED_SAMPLES/CARDS_COURTYARD_B_T/polygons.mat'))
        #
        # X = imread(join(labelled_samples_url, 'CARDS_COURTYARD_B_T/frame_0011.jpg'))

        if memmapped:
            if not fmmap_exists:
                #del X  # flush the file
                utils.talk('flush memmap to file', self.verbose)
                X.flush() # write memmap to files
                del X

            X = np.memmap(fmmap, dtype='uint8', mode='r', shape=X_shape)
        else:
            X = np.array(X, 'uint8')

        return X, np.array(Y)
