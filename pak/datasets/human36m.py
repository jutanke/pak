import numpy as np
from os.path import join, isdir, isfile
from os import listdir
from spacepy import pycdf
import cv2
import h5py


class Human36m:

    def __init__(self, root):
        """

        """
        assert isdir(root), 'cannot find ' + root + ': download human3.6m'
        self.root = root
        self.actors = ["S1", 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
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

        # load cameras
        camera_file = join(root, 'cameras.h5')
        assert isfile(camera_file)
        cam_file = h5py.File(camera_file, 'r')

        self.Calib = []

        for sub in [1, 5, 6, 7, 8, 9, 11]:
            actors = cam_file['subject' + str(sub)]

            Calib_per_actor = []
            self.Calib.append(Calib_per_actor)
            for cid in [1, 2, 3, 4]:
                camera = {}
                cam = actors['camera' + str(cid)]

                name = ''
                for c in cam['Name'][:]:
                    name += chr(c)
                camera['name'] = name

                camera['R'] = cam['R'][:]
                camera['t'] = cam['T'][:]
                camera['c'] = cam['c'][:]
                camera['f'] = cam['f'][:]
                camera['k'] = cam['k'][:]
                camera['p'] = cam['p'][:]
                Calib_per_actor.append(camera)

    def load_videos(self, actor, action, sub_action):
        """ loads video, and memmappes them if not already done
        :param actor:
        :param action:
        :return:
        """
        assert actor in self.actors
        assert action in self.actions
        assert sub_action == 0 or sub_action == 1
        video_dir = join(join(self.root, actor), 'Videos')
        fmap = join(video_dir,
                    'mmap_' + actor + '_' + action + '_' + str(sub_action) + '.npy')

        is_memmaped = isfile(fmap)

        # -- find name --
        # this dataset is so damn messy omg...
        fixed_action = ''

        has_action_with1 = False
        V = [f for f in listdir(video_dir) if f.startswith(action + ' 1')]
        if len(V) == 4:
            has_action_with1 = True
        else:
            assert len(V) == 0

        has_action_with2 = False
        V = [f for f in listdir(video_dir) if f.startswith(action + ' 2')]
        if len(V) == 4:
            has_action_with2 = True
        else:
            assert len(V) == 0

        if has_action_with1 and has_action_with2:
            fixed_action = action + ' 1' if sub_action == 0 else action + ' 2'
        elif has_action_with1:
            fixed_action = action + '.' if sub_action == 0 else action + ' 1'

        # -- end find name --
        videos = sorted(
            [f for f in listdir(video_dir) if f.startswith(fixed_action)])

        if (actor == 'S1' and action == 'Walking') or \
                action == 'Sitting':
            videos = videos[0:4]

        assert len(videos) == 4

        X = None
        shape = None
        H = None
        W = None

        for vid, video in enumerate(videos):
            video = join(video_dir, video)
            print('\tload video ', video)
            Im = []
            cap = cv2.VideoCapture(video)
            while True:
                ret, im = cap.read()
                if not ret:
                    break
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                Im.append(im)

            if X is None:
                if H is None and W is None:
                    H, W, c = Im[0].shape
                    assert c == 3
                shape = (4, len(Im), H, W, 3)
                if not is_memmaped:
                    X = np.memmap(fmap, dtype='uint8', mode='w+',
                                  shape=shape)
                if is_memmaped:
                    # we just needed the shape
                    break

            for i, im in enumerate(Im):
                if i >= shape[1]:  # just skip "extra" frames
                    break
                h, w, c = im.shape
                X[vid, i, 0:min(h, H), 0:min(w, W), :] =\
                    im[0:min(h, H), 0:min(w, W), :]

        assert shape is not None

        if not is_memmaped:
            del X  # flush

        X = np.memmap(fmap, dtype='uint8', mode='r', shape=shape)
        return X

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
        
        if actor != 'S1' and action == 'WalkingDog':
            action = 'WalkDog'
        
        videos = sorted(
            [f for f in listdir(cdf_dir) if f.startswith(action)])

        if (actor == 'S1' and action == 'Walking') or \
                action == 'Sitting':
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


