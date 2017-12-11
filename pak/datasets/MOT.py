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


class MOT_X(Dataset):

    def __init__(self, root, root_export, name, url, verbose=True, resize=None):
        Dataset.__init__(self, name, root, verbose)
        self.root_export = root_export
        self.download_and_unzip(url)
        self.resize = resize


    def get_raw(self, folder, parent, memmapped=False):
        """ get the raw train data
        """
        root = join(self.root_export, parent)
        loc = join(root, folder)

        # X
        img_loc = join(loc, "img1")
        imgs = sorted([join(img_loc, f) \
                for f in listdir(img_loc) if isfile(join(img_loc, f))])
        if memmapped:
            if self.resize is not None:
                raise Exception('Cannot resize and use memmap')

            # we NEED the data shape!
            num_frames = len(imgs)
            w,h,c = imread(imgs[0]).shape
            data_shape = (num_frames, w, h, c)

            fmmap = join(loc, 'data.memmap')

            if not isfile(fmmap):
                # data has to be created
                utils.talk(self.name + ": create memmapped file " + fmmap, self.verbose)
                X = np.memmap(fmmap, dtype='uint8', mode="w+", shape=data_shape)
                for i,f in enumerate(imgs):
                    X[i] = imread(f)

                X.flush()
                del X

            utils.talk(self.name + ": load memmapped file " + fmmap, self.verbose)
            X = np.memmap(fmmap, dtype='uint8', mode='r', shape=data_shape)

        else:
            # Not memmapped files -> load everything into memory
            if self.resize is None:
                X = np.array([imread(f) for f in imgs], 'uint8')
            else:
                X = np.array([imresize(imread(f), size=self.resize) for f in imgs], 'uint8')
        utils.talk(self.name + ' X loaded', self.verbose)

        # Y-det
        det_txt = join(join(loc, "det"), 'det.txt')
        Y_det = np.loadtxt(det_txt, delimiter=',')
        utils.talk(self.name + ' Y_det loaded', self.verbose)

        return X, Y_det

    def get_test(self, folder, memmapped=False):
        """ Gets the raw MOT data for testing
        """
        parent = 'test'
        return MOT_X.get_raw(self, folder, parent=parent, memmapped=memmapped)

    def get_train(self, folder, memmapped=False):
        """ Gets the raw MOT data for training
        """
        parent = 'train'
        X, Y_det = MOT_X.get_raw(self, folder, parent=parent, memmapped=memmapped)

        # Y-gt
        root = join(self.root_export, parent)
        loc = join(root, folder)
        gt_txt = join(join(loc, "gt"), 'gt.txt')
        Y_gt = np.loadtxt(gt_txt, delimiter=',')
        utils.talk(self.name + ' Y_gt loaded', self.verbose)

        return X, Y_det, Y_gt


    def get_test_imgfolder(self, folder):
        return self.get_raw_folder(folder, 'test')


    def get_test_imgfolder(self, folder):
        return self.get_raw_folder(folder, 'train')


    def get_raw_folder(self, folder, parent):
        root = join(self.root_export, parent)
        loc = join(root, folder)
        img_loc = join(loc, "img1")
        assert exists(img_loc)
        return img_loc


    def get_train_folders(self):
        raise NotImplementedError("Must be overriden")


# =========================================
#  MOT16
# =========================================

class MOT16(MOT_X):
    """ MOT16 dataset
    """

    def __init__(self, root, verbose=True, resize=None):
        root_export = join(root, "MOT16")  # force dir name for unzipping
        MOT_X.__init__(self, root, root_export, "MOT16",
                "https://motchallenge.net/data/MOT16.zip", verbose, resize)


    def get_train_folders(self):
        return ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", \
            "MOT16-10", "MOT16-11", "MOT16-13"]

    def get_test_folders(self):
        return ["MOT16-01", "MOT16-03", "MOT16-06", "MOT16-07", \
            "MOT16-08", "MOT16-12", "MOT16-14"]

    def label_id_to_class(self, label_id):
        """ converts the label number id to
            the true label content
        """
        return ["NONE",
                "Pedestrian",
                "Person on vehicle",
                "Car",
                "Bycicle",
                "Motorbike",
                "Non motorized vehicle",
                "Static person",
                "Distractor",
                "Occluder",
                "Occluder on the ground",
                "Occluder full",
                "Reflection" ][int(label_id)]

# =========================================
#  MOT15
# =========================================

class MOT152D(MOT_X):
    """ MOT15 2d dataset
    """

    def __init__(self, root, verbose=True, resize=None):
        MOT_X.__init__(self, root, root, "2DMOT2015",
                "https://motchallenge.net/data/2DMOT2015.zip", verbose, resize)
        self.root_export = join(root, "2DMOT2015")  # the dirs name after unzip


    def get_train_folders(self):
        return ["ADL-Rundle-6", "ETH-Bahnhof", "ETH-Sunnyday", "KITTI-17", \
            "TUD-Campus", "Venice-2", "ADL-Rundle-8", "ETH-Pedcross2", \
            "KITTI-13", "PETS09-S2L1", "TUD-Stadtmitte"]

    def get_test_folders(self):
        return ["ADL-Rundle-1", "ADL-Rundle-3", "AVG-TownCentre", \
            "ETH-Jelmoli", "KITTI-16", "PETS09-S2L2", "Venice-1", \
            "ETH-Crossing", "ETH-Linthescher", "KITTI-19", "TUD-Crossing"]
