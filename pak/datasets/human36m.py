import numpy as np
from os.path import join, isdir, isfile
from os import listdir
from spacepy import pycdf


class Human36m:

    def __init__(self, root):
        """

        """
        assert isdir(root), 'cannot find ' + root + ': download human3.6m'
        self.root = root
        self.actors = ["S1", 'S5', 'S6', 'S7', 'S8', 'S9']
        self.actions = [
            'Directions',
            'Discussion',
            'Eating',
            'Greeting',
            'Phoning',
            'Posing',
            'Purchases',
            'Sitting',
            'SittingDown',
            'Smoking',
            'Photo',
            'Waiting',
            'Walking',
            'WalkingDog',
            'WalkTogether'
        ]

    def get_3d(self, actor, action, sub_action=0):
        """

        :param actor:
        :param action:
        :param sub_action:
        :return:
        """
        assert actor in self.actors
        assert action in self.actions
        assert sub_action == 0 or sub_action == 1
        if actor == 'S1' and action == 'Photo':
            # "fix" messy data labeling
            action = 'TakingPhoto'

        root = self.root
        cdf_dir = join(join(root, actor), 'MyPoseFeatures')
        cdf_dir = join(cdf_dir, 'D3_Positions')

        videos = sorted(
            [f for f in listdir(cdf_dir) if f.startswith(action)])

        if action == 'Walking':
            # separate Walking from WalkingDog
            assert len(videos) == 4
            videos = videos[0:2]

        assert len(videos) == 2, '# of videos:' + str(len(videos))
        a, b = videos
        if len(a) > len(b):  # ['xxx 9.cdf', 'xxx.cdf']
            videos = [b, a]
        else:
            assert len(a) == len(b)

        cdf_file = join(cdf_dir, videos[sub_action])
        assert isfile(cdf_file)
        cdf = pycdf.CDF(cdf_file)

        joints3d = np.squeeze(cdf['Pose']).reshape((-1, 32, 3))
        return joints3d




