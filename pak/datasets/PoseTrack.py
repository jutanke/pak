import numpy as np
from os import listdir
from os.path import isfile, isdir, join


class PoseTrack:

    def __init__(self, posetrack_root):
        """
        :param posetrack_root:
        """
        assert isdir(posetrack_root)
        self.root = posetrack_root
        self.annotation_dir = join(posetrack_root, 'annotations')
        self.train_videos = self.get_all_video_names('train')
        self.test_videos = self.get_all_video_names('test')
        self.val_videos = self.get_all_video_names('val')

    def get_all_video_names(self, type):
        """
        :param type:
        :return:
        """
        assert type in {'train', 'test', 'val'}
        annotation_dir = self.annotation_dir
        loc = join(annotation_dir, type)
        all_files = [f[:-5] for f in sorted(listdir(loc))]
        return all_files



