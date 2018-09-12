import numpy as np
import cv2
from os import makedirs, listdir
from os.path import join, isfile, isdir
from pak.util import download, unzip
from numpy.random import randint
from scipy.io import loadmat


class PennAction_cropped:
    """
        cropped PennAction dataset for Forecasting Human Poses
    """

    def __init__(self, root, verbose=True):
        """

        :param root:
        :param verbose:
        """
        if verbose:
            print('\n**PennAction [cropped]**')

        data_root = join(root, 'pennaction_cropped')
        if not isdir(data_root):
            makedirs(data_root)

        url = 'http://188.138.127.15:81/Datasets/penn-crop.zip'

        data_folder = join(data_root, 'penn-crop')
        if not isdir(data_folder):
            zip_filename = join(data_root, 'penn-crop.zip')
            if not isfile(zip_filename):
                if verbose:
                    print('\tdownload ', url)
                download.download(url, zip_filename)

            if verbose:
                print('\tunzip ', zip_filename)
                unzip.unzip(zip_filename, data_root, verbose=verbose)

        self.data_folder = data_folder
        if verbose:
            print('')

        self.frames_folder = join(data_folder, 'frames')
        labels_folder = join(data_folder, 'labels')
        self.labels_folder = labels_folder
        assert isdir(labels_folder)

        ids = [name[0:4] for name in sorted(listdir(labels_folder))]
        self.ids = ids

        # split train/val
        validation_indices_file = join(data_folder, 'valid_ind.txt')
        assert isfile(validation_indices_file)
        validation_indices = np.loadtxt(validation_indices_file)
        validation_indices = ['%04d' % idx for idx in validation_indices]

        lookup = set(validation_indices)
        self.train_ids = []
        self.val_ids = validation_indices
        for vid in ids:
            if vid not in lookup:
                self.train_ids.append(vid)

        # find the meta-data for each video id

        self.meta = dict()
        for vid in ids:
            vid_labels_file = join(labels_folder, vid + '.mat')
            L = loadmat(vid_labels_file)
            n_frames = L['nframes']
            dimensions = L['dimensions']
            X = np.expand_dims(L['x'], axis=2)
            Y = np.expand_dims(L['y'], axis=2)
            V = np.expand_dims(L['visibility'], axis=2)
            gt = np.concatenate([X, Y, V], axis=2)

            self.meta[vid] = {
                'n_frames': n_frames[0][0],
                'dimensions': dimensions,
                'gt': gt
            }

    def get_random_train_id(self):
        """
        :return: a random valid video id with its length
        """
        ids = self.train_ids
        i = randint(0, len(ids))
        vid = ids[i]
        n_frames = self.meta[vid]['n_frames']
        return vid, n_frames

    def get_random_validation_id(self):
        """
        :return: a random valid video id with its length
        """
        ids = self.val_ids
        i = randint(0, len(ids))
        vid = ids[i]
        n_frames = self.meta[vid]['n_frames']
        return vid, n_frames

    def get_frame(self, vid, frame):
        """
            get a frame from the data
        :param vid:
        :param frame: frame start at 0 for us!!
        :return:
        """
        frame += 1  # but frames start at 1 for the dataset
        frames_folder = self.frames_folder
        dir = join(frames_folder, vid)
        fname = '%06d.jpg' % frame
        fname = join(dir, fname)
        print('fname', fname)
        im = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        gt = self.meta[vid]['gt'][frame]
        return im, gt
