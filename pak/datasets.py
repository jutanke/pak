# Find and download typical datasets for deep learning

import numpy as np#
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

            if not isfile(fzip):
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
            utils.talk(self.root_export + ' found :)', self.verbose)


# =========================================
#  MOT16
# =========================================

class MOT16(Dataset):

    def __init__(self, root, verbose=True, resize=None):
        Dataset.__init__(self, "MOT16", root, verbose)
        url = 'https://motchallenge.net/data/MOT16.zip'
        self.root_export = join(root, "MOT16")
        self.download_and_unzip(url)
        self.resize = resize

    def get_train_impl(self):
        """ impl
        """
        return 1, 2

    def get_train_folders(self):
        return ["MOT16-02", "MOT16-04"]

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

    def get_train_raw(self, folder):
        """ get the raw train data
        """
        root = join(self.root_export, "train")
        loc = join(root, folder)

        # X
        img_loc = join(loc, "img1")
        imgs = sorted([join(img_loc, f) \
                for f in listdir(img_loc) if isfile(join(img_loc, f))])
        if self.resize is None:
            X = np.array([imread(f) for f in imgs], 'uint8')
        else:
            X = np.array([imresize(imread(f), size=self.resize) for f in imgs], 'uint8')
        utils.talk('MOT16 X loaded', self.verbose)

        # Y-det

        det_txt = join(join(loc, "det"), 'det.txt')
        Y_det = np.loadtxt(det_txt, delimiter=',')
        utils.talk('MOT16 Y_det loaded', self.verbose)

        # Y-gt
        gt_txt = join(join(loc, "gt"), 'gt.txt')
        Y_gt = np.loadtxt(gt_txt, delimiter=',')
        utils.talk('MOT16 Y_gt loaded', self.verbose)

        return X, Y_det, Y_gt
