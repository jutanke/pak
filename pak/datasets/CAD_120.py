import numpy as np
import pandas as pd
from os.path import join, isdir, isfile
from os import makedirs

from pak.util import unzip, download


class CAD_120:

    def __init__(self, root):
        """ files
        :param root:
        """
        assert isdir(root)
        data_root = join(root, 'CAD_120')
        if not isdir(data_root):
            makedirs(data_root)
        self.data_root = data_root

        self.actions = sorted([
            'arranging_objects',
            'cleaning_objects',
            'having_meal',
            'making_cereal',
            'microwaving_food',
            'picking_objects',
            'stacking_objects',
            'taking_food',
            'taking_medicine',
            'unstacking_objects'
        ])

        base_url = 'http://pr.cs.cornell.edu/humanactivities/data/'

        self.subjects = [1, 3, 4, 5]

        # map skeleton representation to joint 3d locs only
        self.items = [0]
        for i in range(11, 155, 14):
            for j in range(0, 4):
                self.items.append(i + j)
        self.items = np.array(self.items + list((range(155, 171))))

        # map our reduced joint + conf to actual 3d data
        items3d = []
        for i in range(1, 61, 4):
            items3d += [i + j for j in range(3)]
        self.items3d = np.array(items3d)

        for pid in self.subjects:
            dir_name = 'Subject%01d_annotations' % pid
            dir_loc = join(data_root, dir_name)

            if not isdir(dir_loc):
                zip_loc = join(data_root, dir_name + '.tar.gz')

                if not isfile(zip_loc):
                    print('download ' + dir_name)
                    url = base_url + dir_name + '.tar.gz'
                    download.download(url, zip_loc)

                # unzip folder
                print('unzip ', zip_loc)
                unzip.unzip(zip_loc, dir_loc)

    def plot(self, ax, skel,
             rcolor='orange', lcolor='green', alpha=1, plot_jids=False):
        """
        :param ax:
        :param skel:
        :param rcolor:
        :param lcolor:
        :param alpha:
        :param plot_jids:
        :return:
        """
        assert len(skel) == 61 or len(skel) == 45
        if len(skel) == 61:
            skel = skel[self.items3d]
        skel = skel.reshape((15, 3))
        ax.scatter(*np.transpose(skel), color='gray', alpha=alpha)
        if plot_jids:
            for jid, pt3d in enumerate(skel):
                ax.text(*pt3d, str(jid))

        limbs = np.array([
            (15, 11), (11, 10),
            (14, 9), (9, 8),
            (8, 10),
            (4, 5), (5, 12),
            (6, 7), (7, 13),
            (4, 8), (6, 10),
            (4, 6),
            (3, 4), (3, 6), (2, 4), (2, 6),
            (1, 2)
        ]) - 1

        left_handed = [False, False,
                       True, True,
                       True,
                       True, True,
                       False, False,
                       True, True,
                       True,
                       True, False, True, False,
                       True]
        for is_left, (a, b) in zip(left_handed, limbs):
            color = lcolor if is_left else rcolor
            ax.plot(
                [skel[a][0], skel[b][0]],
                [skel[a][1], skel[b][1]],
                [skel[a][2], skel[b][2]],
                color=color, alpha=alpha
            )

    def get_3d_points_from_skel(self, skel):
        n_frames, n_channels = skel.shape
        assert n_channels == 61, str(skel.shape)
        return skel[:, self.items3d]

    def get_subject(self, pid):
        items = self.items
        items3d = self.items3d
        assert pid in self.subjects
        name = 'Subject%01d_annotations' % pid
        loc = join(join(self.data_root, name), name)
        assert isdir(loc)

        ACTIONS = {}

        for action in self.actions:
            action_loc = join(loc, action)
            assert isdir(action_loc), 'not found:' + action

            label_fname = join(action_loc, 'labeling.txt')
            assert isfile(label_fname)
            labels = pd.read_csv(label_fname,
                                 header=None,
                                 names=list('abcdefghijk'))
            sequences = {}
            sequence_names = list(labels['a'].astype(str))

            ACTIONS[action] = sequences

            # load all sequences:
            for seq_name in set(sequence_names):
                if len(seq_name) == 9:
                    seq_name = '0' + seq_name
                assert len(seq_name) == 10, seq_name
                pts3d_loc = join(action_loc, seq_name + '.txt')
                assert isfile(pts3d_loc), pts3d_loc
                pts3d_with_rot = pd.read_csv(pts3d_loc,
                                             engine='python',
                                             skipfooter=1,).values
                pts3d_w_conf = pts3d_with_rot[:, items]
                n_frames, channels = pts3d_w_conf.shape
                assert channels == 61  # (15 * 4 + 1)

                # transform points
                transform_fname = join(action_loc,
                                       seq_name + '_globalTransform.txt')
                assert isfile(transform_fname)
                P = np.loadtxt(transform_fname, delimiter=',')

                pts3d = pts3d_w_conf[:, items3d]\
                    .reshape((n_frames, 15, 3))
                ones = np.ones((n_frames, 15, 1))
                pts3d_h = np.concatenate([pts3d, ones], axis=2)

                pts3d_h_trans = np.einsum('ijk,kl', pts3d_h, P)
                pts3d_h_trans = pts3d_h_trans / np.expand_dims(
                    pts3d_h_trans[:, :, 3], axis=2
                )

                pts3d_w_conf[:, items3d] = pts3d_h_trans[:, :, 0:3]\
                    .reshape((n_frames, 45))

                assert seq_name not in sequences
                sequences[seq_name] = {
                    'n_frames': n_frames,
                    'skeleton': pts3d_w_conf
                }

            start_frame = list(labels['b'])
            end_frame = list(labels['c'])
            subaction = list(labels['d'])
            subaction = ['None' if pd.isnull(a) else a for a in subaction]

            for seq_name, start, end, subaction in zip(
                sequence_names, start_frame, end_frame, subaction
            ):
                if len(seq_name) == 9:
                    seq_name = '0' + seq_name
                assert seq_name in sequences
                if 'actions' not in sequences[seq_name]:
                    n_frames = sequences[seq_name]['n_frames']
                    actions_per_frame = ['None'] * n_frames
                else:
                    actions_per_frame = sequences[seq_name]['actions']

                for t in range(start-1, end-1):
                    actions_per_frame[t] = subaction

                sequences[seq_name]['actions'] = actions_per_frame

        return ACTIONS

