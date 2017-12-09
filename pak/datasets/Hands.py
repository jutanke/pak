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


class Hand_config(Enum):
    Default = 1
    Square = 2
    AABB = 3

class Hand(Dataset):
    """
    hand dataset (http://www.robots.ox.ac.uk/~vgg/data/hands/)
    """

    def __init__(self, root, verbose=True):
        """ create a hand dataset
        """
        Dataset.__init__(self, "hand_dataset", root, verbose)
        url = 'http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz'
        #self.root_export = join(root, "lspe")
        self.download_and_unzip(url,
            zipfile_name='hand_dataset.tar.gz')
        self.root_export = join(root, "hand_dataset")

    def get_test(self, config=Hand_config.Default):
        return self.get_raw(join('test_dataset', 'test_data'), config)

    def get_train(self, config=Hand_config.Default):
        return self.get_raw(join('training_dataset', 'training_data'),\
            config)

    def get_val(self, config=Hand_config.Default):
        return self.get_raw(join('validation_dataset', 'validation_data'),\
            config)

    def get_raw(self, subfolder, config):
        """ test vs train vs validation

            make_square: if True a 4th point is added to enclose the
                hand
        """
        make_square = config is Hand_config.Square or config is Hand_config.AABB
        aabb = config is Hand_config.AABB

        path = join(self.root_export, subfolder)

        # # annotations
        path_anno = join(path, 'annotations')
        path_imgs = join(path, 'images')

        slates = sorted([splitext(f)[0] for f in listdir(path_imgs) if \
            isfile(join(path_anno, splitext(f)[0] + '.mat')) and \
            isfile(join(path_imgs, splitext(f)[0] + '.jpg'))])

        X = []
        Y = []

        for f in slates:
            img_file = join(path_imgs, f + '.jpg')
            ann_file = join(path_anno, f + '.mat')

            x = imread(img_file)
            X.append(x)

            # --- y ---
            M = loadmat(ann_file)['boxes'][0]
            nbr_boxes = len(M)
            Frame = []
            for i in range(nbr_boxes):
                single_hand = M[i][0][0]
                Hand = []
                for j in range(3):
                    e = single_hand[j][0]
                    Hand.append(np.array((e[1], e[0])))  # first X, then Y ...
                if make_square:
                    v1 = Hand[0]
                    v2 = Hand[1]
                    v3 = Hand[2]

                    # we want to determine {4} given (1),(2),(3)
                    # (1)--(w)--(2)
                    #  .         |
                    #  .        (h)
                    #  .         |
                    # {4}. . . .(3)
                    direction = v3 - v2
                    v4 = v1 + direction
                    Hand.append(v4)

                    if aabb:
                        x1,y1 = v1
                        x2,y2 = v2
                        x3,y3 = v3
                        x4,y4 = v4

                        Hand = [
                            np.array([max([x1,x2,x3,x4]), max([y1,y2,y3,y4])]),
                            np.array([min([x1,x2,x3,x4]), min([y1,y2,y3,y4])])
                        ]


                Frame.append(Hand)
            Y.append(Frame)

        return np.array(X), Y
