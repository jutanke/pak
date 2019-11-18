from os import makedirs
from os.path import join, isfile, isdir
from pak.util import download
from pak.util import unzip
import numpy as np
from scipy.io import loadmat
import cv2


class Shelf:
    """
    Shelf dataset for multiple human multi-view pose estimation
        http://campar.in.tum.de/Chair/MultiHumanPose
    """

    def __init__(self, root, verbose=True):
        """

        :param root:
        :param verbose:
        """
        if verbose:
            print('\n**Shelf dataset**')
        data_root = join(root, 'tum_shelf')
        if not isdir(data_root):
            makedirs(data_root)

        self.verbose = verbose
        self.data_root = data_root

        # download data
        url = 'http://campar.cs.tum.edu/files/belagian/multihuman/Shelf.tar.bz2'
        data_folder = join(data_root, 'Shelf')
        if not isdir(data_folder):
            zip_filename = join(data_root, 'Shelf.tar.bz2')
            if not isfile(zip_filename):
                if verbose:
                    print('\tdownload ' + url)
                download.download(url, zip_filename)

            if verbose:
                print('\nunzip ' + zip_filename)
                unzip.unzip(zip_filename, data_root, verbose)

        if verbose:
            print('\n')

        # load Calibration data
        seq_root = join(data_root, 'Shelf')
        self.seq_root = seq_root
        calibration_dir = join(seq_root, 'Calibration')
        assert isdir(calibration_dir)

        self.Calib = []
        for cam in ['P0.txt', 'P1.txt', 'P2.txt', 'P3.txt', 'P4.txt']:
            fname = join(calibration_dir, cam)
            assert isfile(fname)
            P = np.loadtxt(fname, delimiter=',')
            self.Calib.append(P)

        # GT binary file
        actorsGTmat = join(seq_root, 'actorsGT.mat')
        assert isfile(actorsGTmat)
        M = loadmat(actorsGTmat)
        Actor3d = M['actor3D'][0]
        persons = []
        for pid in range(4):
            pts = []
            Person = Actor3d[pid]
            n = len(Person)
            for frame in range(n):
                pose = Person[frame][0]
                if len(pose) == 1:
                    pts.append(None)
                elif len(pose) == 14:
                    pts.append(pose)
                else:
                    raise ValueError("Weird pose length:" + str(pose))

            persons.append(pts)
        self.Y = persons

    def number_of_frames(self):
        """
        :return: maximum number of frames
        """
        return 3199

    def get_frame(self, frame):
        """
        :param frame: {integer}
        :return:
        """
        assert frame >= 0
        assert frame <= self.number_of_frames()
        seq_root = self.seq_root
        X = np.zeros((5, 776, 1032, 3), 'uint8')
        Y = [
            self.Y[0][frame],
            self.Y[1][frame],
            self.Y[2][frame],
            self.Y[3][frame]
        ]
        for cid in range(5):
            img_dir = join(seq_root, 'Camera' + str(cid))
            assert isdir(img_dir)
            fname = "img_%06d.png" % frame
            fname = join(img_dir, fname)
            assert isfile(fname)
            im = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            X[cid] = im
        return X, Y, self.Calib
