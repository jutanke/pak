from pak.datasets.Dataset import Dataset
import numpy as np
from os import listdir
from os.path import join, isfile, isdir
from scipy.ndimage import imread
from scipy.misc import imresize
from pak import utils
import time


# =========================================
#  MARKET 1501
# =========================================
class Market1501(Dataset):
    """ Market1501 dataset
    """

    def __init__(self, root, verbose=True, force_shape=None, memmapped=True):
        """
        force_shape: (w,h) if set the images will be forced into a
            given shape
        """
        url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
        Dataset.__init__(self, "Market-1501-v15.09.15", root, verbose)
        self.download_and_unzip(url)
        self.root_export = join(root, "Market-1501-v15.09.15")
        # depending on OS / unzip version / full moon the folder might be 'doubly'
        # nested ...
        nested_root = join(self.root_export, "Market-1501-v15.09.15")
        if isdir(nested_root):
            self.root_export = nested_root
        self.memmapped = memmapped

        if force_shape is None:
            self.force_shape = None
        else:
            assert len(force_shape) == 2
            w,h = force_shape
            self.force_shape = (h, w)  # as resize used inversed logic..

    def get_memmapped_file_name(self, folder):
        shape_str = ''
        force_shape = self.force_shape
        if force_shape is not None:
            shape_str = str(force_shape[0]) + 'x' + str(force_shape[1])

        file_name = folder + shape_str  + '.npy'
        return join(self.root_export , file_name)

    def get_memmapped_file_shape(self, folder):
        if folder == 'bounding_box_train':
            n = 12936
        elif folder == 'bounding_box_test':
            n = 19732
        else:
            raise Exception("Cannot find subsystem " + folder)

        force_shape = self.force_shape
        if force_shape is None:
            h, w = 128,64
        else:
            h, w = force_shape

        return n, h, w, 3

    def get_train(self):
        return self.get_raw('bounding_box_train')

    def get_test(self):
        return self.get_raw('bounding_box_test')

    def get_raw(self, folder):
        """ gets the raw identity pictures
        """
        X_is_loaded = False
        if self.memmapped:
            fmmap = self.get_memmapped_file_name(folder)
            data_shape = self.get_memmapped_file_shape(folder)
            if isfile(fmmap):
                X = np.memmap(fmmap, dtype='uint8', mode="r", shape=data_shape)
                X_is_loaded = True

        loc = join(self.root_export, folder)

        imgs = sorted([f for f in listdir(loc) if isfile(join(loc, f)) and \
                    f.endswith('jpg')])
        if not X_is_loaded:
            force_shape = self.force_shape

            if force_shape is None:
                X = np.array([imread(join(loc, f)) for f in imgs], 'uint8')
            else:
                X = np.array([imresize(imread(join(loc, f)), size=force_shape) \
                    for f in imgs], 'uint8')

            if self.memmapped:
                # store to memmap
                X_ = np.memmap(fmmap, dtype='uint8', mode="w+", shape=data_shape)
                for i, x in enumerate(X):
                    X_[i] = x

                utils.talk('flush memmaped ' + fmmap, self.verbose)
                del X_  # flush
                time.sleep(3)

                X = np.memmap(fmmap, dtype='uint8', mode="r", shape=data_shape)

        identities = np.array([int(f[0:2]) if f.startswith('-1') else int(f[0:4]) \
            for f in imgs])
        cameras = np.array([int(f[4]) if f.startswith('-1') else int(f[6]) \
            for f in imgs])
        sequences = np.array([int(f[6]) if f.startswith('-1') else int(f[8]) \
            for f in imgs])
        frames = np.array([int(f[8:14]) if f.startswith('-1') else int(f[10:16]) \
            for f in imgs])

        Y = np.vstack((identities, cameras, sequences, frames))

        return X, np.rollaxis(Y, 1)

    # ----------------------------------------------
    # static methods
    # ----------------------------------------------

    @staticmethod
    def extract_ids(Y):
        """ converts the Y-values into only id values
        """
        return Y[:,0]
