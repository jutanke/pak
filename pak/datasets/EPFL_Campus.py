import numpy as np
from pak.util import download
from pak.util import unzip
from os import makedirs
from os.path import join, isfile, isdir
import cv2
from scipy.io import loadmat


class EPFL_Campus:
    """ EPFL Campus Benchmark
        https://cvlab.epfl.ch/cms/site/cvlab2/lang/en/data/pom
    """

    def __init__(self, root, verbose=True):
        """

        :param root: root location
        :param: verbose: {boolean}
        """
        data_root = join(root, 'epfl_campus')
        if not isdir(data_root):
            makedirs(data_root)

        self.verbose = verbose
        self.data_root = data_root

        # download data:
        seq_root = join(data_root, 'CampusSeq1')
        self.seq_root = seq_root
        if not isdir(seq_root):
            seq_zip = join(data_root, 'CampusSeq1.zip')
            if not isfile(seq_zip):
                url = 'http://188.138.127.15:81/Datasets/CampusSeq1.zip'
                if verbose:
                    print('\ndownload ' + url)
                download.download(url, seq_zip)
            if verbose:
                print('\nunzip ' + seq_zip)
            unzip.unzip(seq_zip, data_root, verbose, del_after_unzip=True)

        # gt is taken from here: http://campar.in.tum.de/Chair/MultiHumanPose
        P0 = np.array([
            [439.06, 180.81, -26.946, 185.95],
            [-5.3416, 88.523, -450.95, 1324],
            [0.0060594, 0.99348, -0.11385, 5.227]
        ])
        P1 = np.array([
            [162.36, -438.34, -17.508, 3347.4],
            [73.3, -10.043, -443.34, 1373.5],
            [0.99035, -0.047887, -0.13009, 6.6849]
        ])
        P2 = np.array([
           [ 237.58, 679.93, -26.772, -1558.3],
           [-43.114, 21.982, -713.6, 1962.8],
           [-0.83557, 0.53325, -0.13216, 11.202]
        ])
        self.Calib = [P0, P1, P2]

        # GT binary file
        actorsGTmat = join(seq_root, 'actorsGT.mat')
        assert isfile(actorsGTmat)
        M = loadmat(actorsGTmat)
        Actor3d = M['actor3D'][0]
        persons = []
        for pid in range(3):
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
        :return: number of frames
        """
        return 2000

    def get_frame(self, frame):
        """

        :param frame: {integer}
        :return:
        """
        seq_root = self.seq_root
        X = np.zeros((3, 288, 360, 3), 'uint8')
        Y = []
        for cid in [0, 1, 2]:
            img_dir = join(seq_root, 'Camera' + str(cid))
            assert isdir(img_dir)
            Y.append(self.Y[cid][frame])  # TODO bug?
            fname = "campus4-c%01d-%05d.png" % (cid, frame)
            fname = join(img_dir, fname)
            assert isfile(fname)

            im = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            X[cid] = im
        return X, Y, self.Calib
