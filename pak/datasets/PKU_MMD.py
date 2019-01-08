import cv2
import numpy as np
import numba as nb
import pandas as pd
from os.path import join, isdir, isfile
from os import makedirs


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

    @staticmethod
    def single_humans_by_pid():
        """ returns a list of lists with all videos
            of single human actions split by actor id
        :return:
        """
        all_pids = [
            ['0002', '0003', '0004', '0057',
             '0058', '0059'],
            ['0007', '0008', '0009', '0010',
             '0101', '0102', '0103', '0104'],
            ['0011', '0012', '0013', '0014',
             '0087', '0088', '0089', '0090'],
            ['0017', '0018', '0019', '0020'],
            ['0021', '0022', '0023', '0024',
             '0051', '0052', '0053', '0054'],
            ['0027', '0028', '0029', '0030'],
            ['0031', '0032', '0033', '0034'],
            ['0037', '0038', '0039', '0040'],
            ['0041', '0042', '0043', '0044',
             '0061', '0062', '0064'],
            ['0047', '0048', '0049', '0050'],
            ['0067', '0068', '0069', '0070'],
            ['0071', '0072', '0073', '0074'],
            ['0077', '0078', '0079', '0080'],
            ['0081', '0082', '0083', '0084'],
            ['0091', '0092', '0093', '0094'],
            ['0097', '0098', '0099', '0100'],
            ['0107', '0108', '0109', '0110'],
            ['0111', '0112', '0113', '0114'],
            ['0117', '0118', '0119', '0120'],
            ['0121', '0122', '0123', '0124'],
            ['0127', '0128', '0129', '0130'],
            ['0131', '0132', '0133', '0134'],
            ['0137', '0138', '0139', '0140'],
            ['0141', '0142', '0143', '0144'],
            ['0147', '0148', '0149', '0150'],
            ['0151', '0152', '0153', '0154'],
            ['0157', '0158', '0159', '0160'],
            ['0161', '0162', '0163', '0164'],
            ['0167', '0168', '0169', '0170'],
            ['0171', '0172', '0173', '0174'],
            ['0177', '0178', '0179', '0180'],
            ['0181', '0182', '0183', '0184'],
            ['0187', '0188', '0189', '0190'],
            ['0191', '0192', '0193', '0194'],
            ['0197', '0198', '0199', '0200'],
            ['0201', '0202', '0203', '0204'],
            ['0207', '0208', '0209', '0210'],
            ['0211', '0212', '0213', '0214'],
            ['0217', '0218', '0219', '0220'],
            ['0221', '0222', '0223', '0224'],
            ['0227', '0228', '0229', '0230'],
            ['0231', '0232', '0233', '0234'],
            ['0237', '0238', '0239', '0240'],
            ['0241', '0242', '0243', '0244'],
            ['0251', '0252', '0253', '0254'],
            ['0257', '0258', '0259', '0260'],
            ['0263', '0264', '0265', '0266'],
            ['0267', '0268', '0269', '0270'],
            ['0273', '0274', '0275', '0276'],
            ['0277', '0278', '0279', '0280'],
            ['0281', '0282', '0283', '0284'],
            ['0287', '0288', '0289', '0290'],
            ['0291', '0292', '0293', '0294'],
            ['0295', '0296', '0297', '0298'],
            ['0301', '0302', '0303', '0304'],
            ['0305', '0306', '0307', '0308'],
            ['0311', '0312', '0313', '0314'],
            ['0315', '0316', '0317', '0318'],
            ['0321', '0322', '0323', '0324'],
            ['0325', '0326', '0327', '0328'],
            ['0331', '0332', '0333', '0334'],
            ['0335', '0336', '0337', '0338'],
            ['0341', '0342', '0343', '0344'],
            ['0345', '0346', '0347', '0348'],
            ['0351', '0352', '0353', '0354'],
            ['0355', '0356', '0357', '0358'],
            ['0361', '0362', '0363', '0364']
        ]

        flat_list = [item for sublist in all_pids for item in sublist]
        assert len(flat_list) == len(set(flat_list))  # no duplicates

        result_pids = []
        for all_vids_pid in all_pids:
            result = []
            for vid in all_vids_pid:
                for cam in ['-L', '-M', '-R']:
                    result.append(vid + cam)
            result_pids.append(result)
        return result_pids

    def __init__(self, root):
        """
        :param root:
        """
        assert isdir(root)
        pku_root = join(root, 'PKUMMD')
        assert isdir(pku_root)
        data_root = join(pku_root, 'Data')
        self.video_root = join(data_root, 'RGB_VIDEO')
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

    def get_frame(self, video, frame):
        """
        :param video: {string} e.g. '0002-L'
        :param frame: {int}
        :return:
        """
        assert len(video) == 6
        img_dir = join(self.video_root, video)
        if not isdir(img_dir):
            print('\tunpacking into ' + img_dir)
            # create images for all frames
            fname = join(self.video_root, video + '.avi')
            assert isfile(fname)
            makedirs(img_dir)
            cap = cv2.VideoCapture(fname)
            ret = True
            frame_nbr = 1
            while ret:
                # Capture frame-by-frame
                ret, im = cap.read()
                img_name = join(img_dir, "frame%05d.png" % frame_nbr)
                cv2.imwrite(img_name, im)
                frame_nbr += 1

        img_name = join(img_dir, "frame%05d.png" % frame)
        im = cv2.imread(img_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

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
