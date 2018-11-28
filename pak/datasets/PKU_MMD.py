import numpy as np
import numba as nb
import pandas as pd
from os.path import join, isdir, isfile


def plot_skeleton(ax, skel,
                  rcolor='orange', lcolor='green', alpha=1):
    """
    :param ax:
    :param skel:
    :param rcolor:
    :param lcolor:
    :param alpha:
    :return:
    """
    ax.scatter(*np.transpose(skel), color='gray', alpha=alpha)

    limbs = np.array([(13, 14), (14, 15), (15, 16),
                      (17, 18), (18, 19), (19, 20),
                      (17, 1), (1, 13),
                      (1, 2), (2, 21), (2, 9), (2, 5),
                      (9, 21), (21, 5), (5, 13), (9, 17),
                      (3, 4), (3, 21),
                      (5, 6), (6, 7), (7, 8), (8, 23), (8, 22),
                      (9, 10), (10, 11), (11, 12), (12, 25), (12, 24)
                      ]) - 1
    left_handed = [True, True, True,
                   False, False, False,
                   False, True,
                   False, False, False, True,
                   False, True, True, False,
                   False, False,
                   True, True, True, True, True,
                   False, False, False, False, False
                   ]
    for is_left, (a, b) in zip(left_handed, limbs):
        color = lcolor if is_left else rcolor
        ax.plot(
            [skel[a][0], skel[b][0]],
            [skel[a][1], skel[b][1]],
            [skel[a][2], skel[b][2]],
            color=color, alpha=alpha
        )


class PKU_MMD:

    def __init__(self, root):
        """
        :param root:
        """
        assert isdir(root)
        pku_root = join(root, 'PKUMMD')
        assert isdir(pku_root)
        data_root = join(pku_root, 'Data')
        data_root = join(data_root, 'PKU_Skeleton_Renew')
        self.data_root = data_root
        label_root = join(pku_root, 'Label')
        label_root = join(label_root, 'Train_Label_PKU_final')
        self.label_root = label_root
        split_root = join(pku_root, 'Split')
        assert isdir(data_root)
        assert isdir(label_root)
        assert isdir(split_root)

        split_file = join(split_root, 'cross-view.txt')
        train_videos = None
        validation_videos = None
        with open(split_file) as f:
            training_line = 1
            validation_line = 3
            for i, line in enumerate(f):
                if i == training_line:
                    train_videos = line.replace(' ', '').split(',')
                if i == validation_line:
                    validation_videos = line.replace(' ', '').split(',')

        self.train_videos = [
            f for f in train_videos if len(f) == 6]
        self.validation_videos = [
            f for f in validation_videos if len(f) == 6]

        # get the names
        fname = join(split_root, 'Actions.xlsx')
        assert isfile(fname)
        xlsx = pd.read_excel(fname)
        action_names = list(xlsx['Action'])
        action_ids = list(xlsx['Label'])

        # ids -> action
        self.action_id_to_action_name = {}
        # action -> id
        self.action_name_to_action_id = {}

        for aid, name in zip(action_ids, action_names):
            self.action_id_to_action_name[aid] = name
            self.action_name_to_action_id[name] = aid

    def get_3d(self, video):
        """
        :param video: {string} e.g. '0002-L'
        :return:
        """
        assert len(video) == 6
        fname = join(self.data_root, video + '.txt')
        assert isfile(fname)

        skeletons = np.loadtxt(fname)
        n_frames, channels = skeletons.shape
        assert channels == 150  # 3 x 25 x 2

        skel1 = skeletons[:, :75].reshape((n_frames, 25, 3))
        skel2 = skeletons[:, 75:].reshape((n_frames, 25, 3))

        skel1 = flip_axis_so_that_z_is_height(skel1)
        skel2 = flip_axis_so_that_z_is_height(skel2)

        # load labels
        y = np.zeros((n_frames, ), np.int32)
        fname = join(self.label_root, video + '.txt')
        assert isfile(fname)
        labels = np.loadtxt(fname, delimiter=',', dtype=np.int32)
        for activity, start_frame, end_frame, _ in labels:
            y[start_frame:end_frame] = activity

        return skel1, skel2, y


@nb.jit(nb.float64[:, :, :](
    nb.float64[:, :, :]
), nopython=True, nogil=True)
def flip_axis_so_that_z_is_height(skel):
    """
    :param skel: (n_frames, 25, 3)
    :return:
    """
    M = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], np.float64)
    result = np.empty(skel.shape, np.float64)
    n_frames = len(skel)
    for t in range(n_frames):
        result[t] = np.transpose(M @ np.transpose(skel[t]))
    return result
