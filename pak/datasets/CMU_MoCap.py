from os.path import isfile, isdir, join
from os import listdir, makedirs
from pak.util import download, unzip
import c3d
import numpy as np


class CMU_MoCap:

    def __init__(self, data_root):
        assert isdir(data_root)

        root = join(data_root, 'cmu_mocap')
        if not isdir(root):
            makedirs(root)

        subject_folder = join(root, 'subjects')
        if not isdir(subject_folder):
            print("[CMU MoCap] download files")

            zip_files = [
                'allc3d_0.zip',
                'allc3d_1a.zip',
                'allc3d_1b.zip',
                'allc3d_234.zip',
                'allc3d_56789.zip'
            ]

            for zip_name in zip_files:
                url = 'http://mocap.cs.cmu.edu/' + zip_name
                zip_file = join(root, zip_name)
                if not isfile(zip_file):
                    print('\t[downloading] ', url)
                    download.download(url, zip_file)
                print('\t[unzipping] ', zip_file)
                unzip.unzip(zip_file, root)

        self.subjects = sorted(listdir(subject_folder))
        self.subject_folder = subject_folder

    def get(self, subject, action):
        subject_loc = join(self.subject_folder, subject)
        assert isdir(subject_loc), subject_loc
        data_file = join(subject_loc, subject + '_' + action + '.c3d')
        assert isfile(data_file), data_file

        with open(data_file, 'rb') as handle:
            reader = c3d.Reader(handle)
            frames = []
            joints = []
            for i, (frame, markers, _) in enumerate(reader.read_frames()):
                frames.append(frame)
                joints.append(markers)

        return np.array(frames), np.array(joints)
