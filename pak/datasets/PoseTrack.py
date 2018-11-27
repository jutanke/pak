import numpy as np
import json
from os import listdir
from os.path import isfile, isdir, join
from pycocotools.coco import COCO


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
        self.annotations_train = None
        self.annotations_val = None
        self.annotations_test = None

    def get_all_video_names(self, type):
        """
        :param type:
        :return:
        """
        loc = self.get_annotation_dir(type)
        all_files = [f[:-5] for f in sorted(listdir(loc))]
        return all_files

    def get_annotation_dir(self, type):
        """ yields the directory of the annotations
        :param type:
        :return:
        """
        assert type in {'train', 'test', 'val'}
        annotation_dir = self.annotation_dir
        loc = join(annotation_dir, type)
        assert isdir(loc)
        return loc

    def get_annotations(self, type):
        """ gets all annotations
        :param type:
        :return:
        """
        loc = self.get_annotation_dir(type)
        all_annotations = self.get_all_video_names(type)
        results = []
        for video_name in all_annotations:
            file_name = join(loc, video_name + '.json')
            with open(file_name) as f:
                data = json.load(f)
            results.append(data)
        return results
