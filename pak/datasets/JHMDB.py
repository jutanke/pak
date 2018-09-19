import numpy as np
import cv2
from os.path import isdir, isfile, join
from os import listdir


class JHMDB:

    def __init__(self, root, fmmap, verbose=False):
        """

        :param root:
        """
        assert isdir(root)
        images_root = join(root, 'images')
        assert isdir(images_root)
        person_poses = join(root, 'person_poses')
        assert isdir(person_poses)

        action_list = sorted(listdir(images_root))
        assert len(action_list) == 21

        self.videos_by_action = {}

        is_memmapped = isfile(fmmap)
        if verbose:
            print('[JHMDB] is memory-mapped:', is_memmapped)
        shape = (32173, 240, 320, 3)
        if not is_memmapped:
            X = np.memmap(fmmap, dtype='uint8', mode='w+', shape=shape)

        i = 0

        for action in action_list:
            self.videos_by_action[action] = {}
            action_root = join(images_root, action)
            assert isdir(action_root)

            if verbose:
                print('-------------------')
                print('[JHMDB]\thandle ', action)
                print('-------------------')

            for vname in sorted(listdir(action_root)):
                if verbose:
                    print('[JHMDB]\t\tload ', vname)
                video_root = join(action_root, vname)
                assert isdir(video_root)
                video_files = sorted([
                    join(video_root, f) for f in listdir(video_root)
                ])

                n_frames = len(video_files)
                start_i = i
                end_i = i + n_frames

                # ground truth
                Y = []
                gt_loc = join(person_poses, vname)
                assert isdir(gt_loc)
                assert len(listdir(gt_loc)) == n_frames
                for frame in range(n_frames):
                    fname = '%05d.txt' % (frame + 1)
                    gt_file = join(gt_loc, fname)
                    y = np.loadtxt(gt_file)
                    Y.append(y)

                self.videos_by_action[action][vname] = (
                    start_i, end_i, np.array(Y)
                )

                if not is_memmapped:
                    video = np.array([
                        cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) \
                        for f in video_files
                    ], 'uint8')
                    for im in video:
                        assert i < 32173
                        X[i, :, :, :] = im
                        i += 1
                else:
                    i += len(video_files)

        if not is_memmapped:
            del X  # flush

        assert i == 32173
        self.X = np.memmap(fmmap, dtype='uint8', mode='r', shape=shape)

    def get_all_actions(self):
        """
        :return: all actions
        """
        return sorted(self.videos_by_action.keys())

    def get_all_videos_for_action(self, action):
        """
        :param action:
        :return: all videos for the given action
        """
        return sorted(self.videos_by_action[action].keys())

    def load(self, action, video):
        """
        Load the video of the given action
        :param action:
        :param video:
        :return:
        """
        start_i, end_i, gt = self.videos_by_action[action][video]
        video = self.X[start_i:end_i, :, :, :]
        return video, gt
