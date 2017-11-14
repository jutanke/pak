# Find and download typical datasets for deep learning

import numpy as np
import zipfile
import urllib.request
import shutil
from os import makedirs, listdir
from os.path import join, isfile, exists
from scipy.ndimage import imread
from scipy.misc import imresize
from pak import utils


class Dataset:
    """ Dataset base class
    """

    def __init__(self, name, root, verbose=True):
        self.name = name
        self.root = root
        self.root_export = root
        self.verbose = verbose
        if not exists(root):
            makedirs(root)

    def get_train(self):
        """ returns the train dataset X and Y
        """
        X, Y = self.get_train_impl()
        return X, Y


    def get_train_impl(self):
        raise NotImplementedError("Must be overriden")


    def download_and_unzip(self, url):
        """ Downloads and unzips a zipped data file

        """
        dest = join(self.root, self.name)
        if not exists(dest):
            utils.talk("could not find folder " + dest + "...", self.verbose)
            fzip = join(self.root, self.name + ".zip")

            if isfile(fzip):
                utils.talk('found ' + fzip)
            else:
                utils.talk("could not find file " + fzip, self.verbose)
                utils.talk("download from " + url, self.verbose)
                with urllib.request.urlopen(url) as res, open(fzip, 'wb') as f:
                    utils.talk(url + " downloaded..", self.verbose)
                    shutil.copyfileobj(res, f)
            zip_ref = zipfile.ZipFile(fzip, 'r')
            utils.talk("unzip " + fzip + " -> " + self.root, self.verbose)
            zip_ref.extractall(self.root_export)
            zip_ref.close()
        else:
            utils.talk(dest + ' found :)', self.verbose)


# =========================================
#  MOTXX
# =========================================

class MOT_X(Dataset):

    def __init__(self, root, root_export, name, url, verbose=True, resize=None):
        Dataset.__init__(self, name, root, verbose)
        self.root_export = root_export
        self.download_and_unzip(url)
        self.resize = resize

    def get_train_impl(self):
        """ impl
        """
        return 1, 2

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

    def get_raw(self, folder, parent):
        """ get the raw train data
        """
        root = join(self.root_export, parent)
        loc = join(root, folder)

        # X
        img_loc = join(loc, "img1")
        imgs = sorted([join(img_loc, f) \
                for f in listdir(img_loc) if isfile(join(img_loc, f))])
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

    def get_test_raw(self, folder):
        """ Gets the raw MOT data for testing
        """
        parent = 'test'
        return MOT_X.get_raw(self, folder, parent=parent)

    def get_train_raw(self, folder):
        """ Gets the raw MOT data for training
        """
        parent = 'train'
        X, Y_det = MOT_X.get_raw(self, folder, parent=parent)

        # Y-gt
        root = join(self.root_export, parent)
        loc = join(root, folder)
        gt_txt = join(join(loc, "gt"), 'gt.txt')
        Y_gt = np.loadtxt(gt_txt, delimiter=',')
        utils.talk(self.name + ' Y_gt loaded', self.verbose)

        return X, Y_det, Y_gt


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
        return ["MOT16-02", "MOT16-04"]


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


# =========================================
#  MARKET 1501
# =========================================

class Market1501(Dataset):
    """ Market1501 dataset
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, "Market-1501-v15.09.15", root, verbose)
        self.download_and_unzip('NOT-YET-THERE')
        self.root_export = join(root, "Market-1501-v15.09.15")

    def get_train_raw(self):
        return self.get_raw('bounding_box_train')

    def get_test_raw(self):
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

        return X, Y



# =========================================
#  CUHK03
# =========================================

class cuhk03(Dataset):
    """ cuhk03 dataset
    """

    def __init__(self, root):
        Dataset.__init__(self, "cuhk03_release", root, verbose)
