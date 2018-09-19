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
                if is_memmapped:
                    continue
                if verbose:
                    print('[JHMDB]\t\tload ', vname)
                video_root = join(action_root, vname)
                assert isdir(video_root)
                video_files = sorted([
                    join(video_root, f) for f in listdir(video_root)
                ])

                start_i = i
                end_i = i + len(video_files)

                self.videos_by_action[action][vname] = (
                    start_i, end_i
                )

                video = np.array([
                    cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) \
                    for f in video_files
                ], 'uint8')
                for im in video:
                    assert i < 32173
                    X[i, :, :, :] = im
                    i += 1

        if not is_memmapped:
            assert i == 32173
            del X  # flush

        self.X = np.memmap(fmmap, dtype='uint8', mode='r', shape=shape)