from pak.datasets.Dataset import Dataset
import numpy as np
from pak import utils
from pak.util import download
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
import time
import cv2


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

        # gt is taken from here: http://campar.in.tum.de/Chair/MultiHumanPose
        self.P0 = np.array([
            [439.06, 180.81, -26.946, 185.95],
            [-5.3416, 88.523, -450.95, 1324],
            [0.0060594, 0.99348, -0.11385, 5.227]
        ])
        self.P1 = np.array([
            [162.36, -438.34, -17.508, 3347.4],
            [73.3, -10.043, -443.34, 1373.5],
            [0.99035, -0.047887, -0.13009, 6.6849]
        ])
        self.P2 = np.array([
           [ 237.58, 679.93, -26.772, -1558.3],
           [-43.114, 21.982, -713.6, 1962.8],
           [-0.83557, 0.53325, -0.13216, 11.202]
        ])

    def get_sequence(self, seq_nbr):
        """

        :param seq_nbr: can be 1 or 2
        :return:
        """
        assert seq_nbr in [1, 2], "invalid seq nbr:" + str(seq_nbr)
        data_root = self.data_root
        verbose = self.verbose
        if verbose:
            print("[EPFL Campus] get sequence " + str(seq_nbr))
        seq_root = join(data_root, 'Seq' + str(seq_nbr))
        if not isdir(seq_root):
            makedirs(seq_root)

        url_root = 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/'
        fname_base = 'campus4-c' if seq_nbr == 1 else 'campus7-c'

        X = np.zeros((3, 1720, 288, 360, 3), 'uint8')
        for cid in [0, 1, 2]:
            if seq_nbr == 2 and cid > 0:
                # this is a bug on their website...
                url_root = 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/'
            fname = fname_base + str(cid) + '.avi'
            url = url_root + fname
            f_loc = join(seq_root, fname)
            if not isfile(f_loc):
                if verbose:
                    print("\tdownload " + fname)
                download.download(url, f_loc)

            if verbose:
                print('\tload ' + fname)

            cap = cv2.VideoCapture(f_loc)
            cur_frame = 0
            while True:
                valid, frame = cap.read()
                if not valid:
                    break
                if cur_frame == 1720:
                    # as the videos have different
                    # length we cut off the longer
                    # ones...
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                X[cid, cur_frame] = frame

                cur_frame += 1

            if verbose:
                print('\tfinish loading ' + fname)

        return X

