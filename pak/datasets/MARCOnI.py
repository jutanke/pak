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
import h5py
from enum import Enum

class MARCOnI(Dataset):
    """ MPII http://marconi.mpi-inf.mpg.de/#download
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, 'marconi', root, verbose)
        utils.talk('(MARCOnl)', verbose)
        self._verbose = verbose
        self.root_export = join(root, 'marconi')
        self.download_set("Soccer")
        self.download_set("Kickbox")
        self.download_set("SBoard")
        self.download_set("Soccer2")
        self.download_set("Walk1")
        self.download_set("Walk2")
        self.download_set("Volleyball")
        self.download_set("Juggling")
        self.download_set("Run2")


    def get_memmapped_file_names(self, name):
        """ get the file names for the memmaped files
        """
        root_dir = join(self.root, 'marconi/' + name)
        fImages = join(root_dir, 'Images.npy')
        fCNNs = join(root_dir, 'CNN_Detections.npy')
        fAnnot = join(root_dir, 'Annotations.npy')
        return fImages, fCNNs, fAnnot


    def __getitem__(self, name):
        """ [ access ]
        """
        fImages, fCNNs, fAnnot = self.get_memmapped_file_names(name)
        self.create_memmapped_data(name)

        shape = self.get_video_shape(name)
        X = np.memmap(fImages, dtype='uint8', mode='r', shape=shape)

        n, m, h, w, _ = shape
        CNN = np.memmap(fCNNs, dtype='uint8', mode='r', shape=
            (n, m, int(h/2), int(w/2), 14))

        return X, CNN


    def get_video_shape(self, name):
        """ get the data shape for the given data
        """
        root_dir = join(self.root, 'marconi/' + name)
        image_dir = join(root_dir, join('Images', 'Images'))
        assert isdir(image_dir), 'Image directory must exist ->' + image_dir

        Cam_Ids = sorted(list(
            set([f[6:8] for f in listdir(image_dir) if f.endswith('.jpg')])))

        Camera_Images = []
        m = -1  # number of frames per camera
        w, h = -1, -1  # width/height of videos
        for cam_id in Cam_Ids:
            Images = sorted([join(image_dir, f) for f in listdir(image_dir) if \
                f.endswith('.jpg') and f.startswith('frame_' + cam_id)])
            Camera_Images.append(Images)
            if m < 0:
                m = len(Images)
            else:
                assert m == len(Images)

            I = imread(Images[0])
            h_, w_, _ = I.shape
            if w < 0 and h < 0:
                w, h = w_, h_
            else:
                assert w == w_ and h == h_

        n = len(Cam_Ids)  # number of cameras
        return (n, m, h, w, 3)


    def create_memmapped_data(self, name):
        """ create the memory-mapped dataset for the given sub-set
        """
        root_dir = join(self.root, 'marconi/' + name)
        assert isdir(root_dir), 'directory ' + root_dir + " must exist!"

        fImages, fCNNs, fAnnot = self.get_memmapped_file_names(name)

        # ----------------------------
        #     memory-mapped video
        # ----------------------------
        if not isfile(fImages):
            image_dir = join(root_dir, join('Images', 'Images'))
            assert isdir(image_dir), 'Image directory must exist ->' + image_dir

            Cam_Ids = sorted(list(
                set([f[6:8] for f in listdir(image_dir) if f.endswith('.jpg')])))

            Camera_Images = []
            m = -1  # number of frames per camera
            w, h = -1, -1  # width/height of videos
            for cam_id in Cam_Ids:
                Images = sorted([join(image_dir, f) for f in listdir(image_dir) if \
                    f.endswith('.jpg') and f.startswith('frame_' + cam_id)])
                Camera_Images.append(Images)
                if m < 0:
                    m = len(Images)
                else:
                    assert m == len(Images)

                I = imread(Images[0])
                h_, w_, _ = I.shape
                if w < 0 and h < 0:
                    w, h = w_, h_
                else:
                    assert w == w_ and h == h_

            n = len(Cam_Ids)  # number of cameras

            X = np.memmap(fImages, dtype='uint8', mode='w+', shape=(n, m, h, w, 3))

            for c, Images in enumerate(Camera_Images):
                I = np.array([imread(f) for f in Images], 'uint8')
                X[c] = I

            del X
        else:
            n, m, h, w, _ = self.get_video_shape(name)


        # ----------------------------
        #     CNN detections
        # ----------------------------
        if not isfile(fCNNs):
            cnn_dir = join(root_dir, join('CNN_Detections', 'CNN_Detections'))
            assert isdir(cnn_dir), 'CNN-Detection folder must exist ->' + cnn_dir

            Cam_Ids_cnn = sorted(list(
                set([f[0:8] for f in listdir(cnn_dir) if f.endswith('.png')])))
            assert len(Cam_Ids_cnn) == n

            joints = ["j%03d.png" % (j,) for j in range(1, 15)]

            CNNs = []
            for cam_id in Cam_Ids_cnn:
                Joints = []
                for j in joints:
                    J = sorted([join(cnn_dir, f) for f in listdir(cnn_dir) if \
                        f.startswith(cam_id) and f.endswith(j)])
                    assert len(J) == m
                    Joints.append(J)

                CNNs.append(Joints)

            X = np.memmap(fCNNs, dtype='uint8', mode='w+', shape=
                (n, m, int(h/2), int(w/2), 14))

            for c, cnn in enumerate(CNNs):
                for j, J in enumerate(cnn):
                    J = np.array([imread(f) for f in J], 'uint8')
                    X[c,:,:,:,j] = J

            del X


    def download_set(self, name):
        """ download the given set
        """
        self.root_export = join(self.root, 'marconi/' + name)
        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + '/Images.zip'
        self.verbose = False
        self.download_and_unzip(url, zipfile_name='Images.zip',
            dest_folder='marconi/' + name + "/Images", dest_force=True,
            root_folder='marconi/' + name)

        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + '/CNN_Detections.zip'
        self.download_and_unzip(url, zipfile_name='CNN_Detections.zip',
            dest_folder='marconi/' + name + "/CNN_Detections", dest_force=True,
            root_folder='marconi/' + name)

        self.root_export = join(self.root, 'marconi')
        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + \
            '/Annotations.mat'
        self.download_file(url, file_name='Annotations.mat', dest_folder=name)

        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + \
            '/Calibration.dat'
        self.download_file(url, file_name='Calibration.dat', dest_folder=name)
        self.verbose = self._verbose
        utils.talk("\t" + name + " finished", self._verbose)
