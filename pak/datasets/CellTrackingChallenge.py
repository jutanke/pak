import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from pak.util import unzip
from scipy.ndimage import imread
import pak.utils as utils
from pak.util import download, unzip
import cv2


class CellTrackingChallenge2D:
    """  Cell-Tracking Challenge
         http://www.celltrackingchallenge.net/index.html
    """

    def __init__(self, root, username, pw):
        """
        :param root:
        :param username: to access the data a username and
        :param pw:  password are required
        """
        data_root = join(root, 'celltrackingchallenge')
        self.data_root = data_root
        test_root = join(data_root, 'test')
        train_root = join(data_root, 'train')

        if not isdir(test_root):
            makedirs(test_root)
        if not isdir(train_root):
            makedirs(train_root)

        Train_urls = CellTrackingChallenge2D.get_training_dataset_urls()
        Test_urls = CellTrackingChallenge2D.get_test_dataset_urls()

        for url_train, url_test in zip(Train_urls, Test_urls):
            fname = utils.get_filename_from_url(url_test)
            assert fname == utils.get_filename_from_url(url_train)
            fzip_train = join(train_root, fname)
            fzip_test = join(test_root, fname)
            if not isfile(fzip_train):
                utils.talk("could not find train file:" + fname + ", .. downloading", True)
                download.download_with_login(url_train, train_root, username, pw)
            if not isfile(fzip_test):
                utils.talk("could not find test file:" + fname + ", .. downloading", True)
                download.download_with_login(url_test, test_root, username, pw)

            data_name = fname[:-4]
            cur_train_loc = join(train_root, data_name)
            cur_test_loc = join(test_root, data_name)
            if not isdir(cur_train_loc):
                utils.talk("Could not find folder " + cur_train_loc + ', .. unzip', True)
                unzip.unzip(fzip_train, train_root)
            if not isdir(cur_test_loc):
                utils.talk("Could not find folder " + cur_test_loc + ', .. unzip', True)
                unzip.unzip(fzip_test, test_root)

    def get_train(self, name):
        """
        get the training set for the given dataset
        :param name: of the datase
        :return: segmentation and tracking
        """
        root = join(self.data_root, 'train')
        loc = join(root, name)
        assert isdir(loc), 'dir ' + loc + ' must exist'
        for dset in ['01', '02']:
            dset_loc = join(loc, dset); assert isdir(dset_loc)
            dset_gt_loc = join(loc, dset + '_GT'); assert isdir(dset_gt_loc)

            Imgs = []
            for f in sorted([f for f in listdir(dset_loc) if f.endswith('.tif')]):
                #I = imread(join(dset_loc, f), mode='L')
                I = cv2.imread(join(dset_loc, f), 0)
                Imgs.append(I)

            # get segmentation
            seg_loc = join(dset_gt_loc, 'SEG'); assert isdir(seg_loc)
            # not all frames are segmented!
            segmented = sorted([f for f in listdir(seg_loc) if f.endswith('.tif')])

            Seg_frames = []
            Segs = []
            for seg in segmented:
                frame = int(seg[7:10]); Seg_frames.append(frame)
                S = cv2.imread(join(seg_loc, seg), cv2.IMREAD_ANYDEPTH)
                Segs.append(S)

        return np.array(Imgs, 'uint8'), np.array(Segs), Seg_frames

    def get_train_all_segmented(self):
        """
        :return: only the data with segmented images
        """
        Imgs = []
        Segs = []
        for ds in CellTrackingChallenge2D.get_dataset_names():
            X, Seg, Seg_idx = self.get_train(ds)
            for S, i in zip(Seg, Seg_idx):
                Imgs.append(X[i])
                Segs.append(S)

            del X; del Seg; del Seg_idx

        return Imgs, Segs


    # --- static ---

    @staticmethod
    def get_dataset_names():
        return [
            'Fluo-C2DL-MSC',
            'Fluo-N2DH-GOWT1',
            'Fluo-N2DL-HeLa',
            'DIC-C2DH-HeLa',
            'PhC-C2DL-PSC',
            'Fluo-N2DH-SIM+'
        ]

    @staticmethod
    def get_training_dataset_urls():
        return [
            'http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip',
            'http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip',
            'http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip',
            #'http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip',
            'http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip',
            'http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip',
            'http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip'
        ]

    @staticmethod
    def get_test_dataset_urls():
        return [
            'http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C2DL-MSC.zip',
            'http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-GOWT1.zip',
            'http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DL-HeLa.zip',
            #'http://data.celltrackingchallenge.net/challenge-datasets/PhC-C2DH-U373.zip',
            'http://data.celltrackingchallenge.net/challenge-datasets/DIC-C2DH-HeLa.zip',
            'http://data.celltrackingchallenge.net/challenge-datasets/PhC-C2DL-PSC.zip',
            'http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-SIM+.zip'
        ]