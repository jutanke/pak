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

        # define what joints connect as limbs
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1

        # define what joints are left- and which ones are right
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    def get_cdf_file(self, type, actor, action, sub_action):
        """ helper function to fuse together the string to
            find the cdf file
        :param type:
        :param actor:
        :param action:
        :param sub_action:
        :return:
        """
        assert type in {'D3_Positions', 'D3_Angles', 'RawAngles'}
        assert actor in self.actors
        assert action in self.actions
        assert sub_action == 0 or sub_action == 1
        if actor == 'S1' and action == 'Photo':
            # "fix" messy data labeling
            action = 'TakingPhoto'

        root = self.root
        cdf_dir = join(join(root, actor), 'MyPoseFeatures')
        cdf_dir = join(cdf_dir, type)

        videos = sorted(
            [f for f in listdir(cdf_dir) if f.startswith(action)])

        if action == 'Walking' or action == 'Sitting':
            # separate Walking from WalkingDog OR
            # separate Sitting from SittingDown
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
        return cdf_file

    def get_3d(self, actor, action, sub_action=0):
        """
        :param actor:
        :param action:
        :param sub_action:
        :return:
        """
        cdf_file = self.get_cdf_file('D3_Positions',
                                     actor, action, sub_action)
        cdf = pycdf.CDF(cdf_file)

        joints3d = np.squeeze(cdf['Pose']).reshape((-1, 32, 3))
        return joints3d

    def get_3d_angles(self, actor, action, sub_action=0):
        """
        :param actor:
        :param action:
        :param sub_action:
        :return:
        """
        cdf_file = self.get_cdf_file('D3_Angles',
                                     actor, action, sub_action)
        cdf = pycdf.CDF(cdf_file)
        angles3d = np.squeeze(cdf['Pose'])
        return angles3d

    def get_raw_angles(self, actor, action, sub_action=0):
        """
        :param actor:
        :param action:
        :param sub_action:
        :return:
        """
        cdf_file = self.get_cdf_file('RawAngles',
                                     actor, action, sub_action)
        cdf = pycdf.CDF(cdf_file)
        angles3d = np.squeeze(cdf['Pose'])
        return angles3d

    @staticmethod
    def plot_human3d(ax, human3d,
                     plot_only_limbs=True, plot_jid=True,
                     lcolor="#3498db", rcolor="#e74c3c", alpha=0.5):
        """ plots a human onto a subplot
        :param ax: subplot
        :param human3d: [32 x 3]
        :param plot_only_limbs plot only joints that are on a limb
        :param plot_jid if True plots the jid next to the 3d location
        :param lcolor:
        :param rcolor:
        :param alpha:
        :return:
        """
        assert len(human3d.shape) == 2
        n_joints, c = human3d.shape
        assert n_joints == 32
        assert c == 3
        I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1

        # define what joints are left- and which ones are right
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

        plots = []
        valid_3d_ids = set(I).union(set(J))

        vals = np.zeros((32, 3))
        for i in np.arange(len(I)):
            x = np.array([vals[I[i], 0], vals[J[i], 0]])
            y = np.array([vals[I[i], 1], vals[J[i], 1]])
            z = np.array([vals[I[i], 2], vals[J[i], 2]])
            plots.append(ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor, alpha=alpha))

        vals = human3d
        for i in np.arange(len(I)):
            x = np.array([vals[I[i], 0], vals[J[i], 0]])
            y = np.array([vals[I[i], 1], vals[J[i], 1]])
            z = np.array([vals[I[i], 2], vals[J[i], 2]])
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)

        for jid, (x, y, z) in enumerate(human3d):
            if plot_only_limbs and jid not in valid_3d_ids:
                continue
            c = 'green' if jid in valid_3d_ids else 'red'
            ax.scatter(x, y, z, color=c, alpha=alpha)
            if plot_jid:
                ax.text(x, y, z, str(jid))


