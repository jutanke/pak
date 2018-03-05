from pak.datasets.Dataset import Dataset
import numpy as np
from pak import utils
from pak.util import download, unzip
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext


class UMPM:
    """ Utrecht Multi-Person Motion Benchmark
        http://www.projects.science.uu.nl/umpm/data/data.shtml
    """

    def __init__(self, root, username, password, verbose=True):
        """
        """
        utils.talk("UMPM", verbose)

        data_root = join(root, 'umpm')
        root_url = 'http://umpm-mirror.cs.uu.nl/download/'

        if not isdir(data_root):
            makedirs(data_root)

        for file in UMPM.get_file_list():
            cur_loc = join(data_root, file)
            fzip = join(cur_loc, file + ".zip")
            cur_url = root_url + file + '.zip'

            print("ZIP:", isfile(fzip))

            # fc3d_gt = join(data_root, file + '.c3d')
            # cur_url_gt = root_url_gt + file + '.c3d'

            if not isdir(cur_loc):
                utils.talk("could not find location " + file, verbose)

                if not isfile(fzip):
                    utils.talk("could not find file " + file + '.zip', verbose)
                    download.download_with_login(
                        cur_url,
                        cur_loc,
                        username,
                        password)

            if not isdir(join(cur_loc, 'Groundtruth')):
                # is not unzipped
                utils.talk("unzipping " + fzip, verbose)
                unzip.unzip(fzip, cur_loc)

            # if not isfile(fc3d_gt):
            #     utils.talk("could not find c3d file " + file, verbose)
            #     download.download_with_login(cur_url_gt, cur_loc, username, password)


    # -------- static --------

    @staticmethod
    def get_file_list():
        """
        :return: all the files stored on the server
        """
        return [
            'p1_chair_2',
            'p1_grab_3',
            'p1_orthosyn_1',
            'p1_table_2',
            'p1_triangle_1',
            'p2_ball_1',
            'p2_chair_1',
            'p2_chair_2',
            'p2_circle_01',
            'p2_free_1',
            'p2_free_2',
            'p2_grab_1',
            'p2_grab_2',
            'p2_meet_1',
            'p2_orthosyn_1',
            'p2_staticsyn_1',
            'p2_table_1',
            'p2_table_2',
            'p3_ball_12',
            'p3_ball_1',
            'p3_ball_2',
            'p3_ball_3',
            'p3_chair_11',
            'p3_chair_12',
            'p3_chair_15',
            'p3_chair_16',
            'p3_chair_1',
            'p3_chair_2',
            'p3_circle_15',
            'p3_circle_16',
            'p3_circle_2',
            'p3_circlesyn_11',
            'p3_free_11',
            'p3_free_12',
            'p3_free_1',
            'p3_free_2',
            'p3_grab_11',
            'p3_grab_12',
            'p3_grab_1',
            'p3_grab_2',
            'p3_meet_11',
            'p3_meet_12',
            'p3_meet_1',
            'p3_meet_2',
            'p3_orthosyn_11',
            'p3_orthosyn_12',
            'p3_orthosyn_2',
            'p3_staticsyn_11',
            'p3_staticsyn_12',
            'p3_staticsyn_2',
            'p3_table_11',
            'p3_table_2',
            'p3_triangle_11',
            'p3_triangle_12',
            'p3_triangle_13',
            'p3_triangle_1',
            'p3_triangle_2',
            'p3_triangle_3',
            'p4_ball_11',
            'p4_ball_12',
            'p4_chair_11',
            'p4_chair_1',
            'p4_circle_11',
            'p4_circle_12',
            'p4_free_11',
            'p4_free_1',
            'p4_grab_11',
            'p4_grab_1',
            'p4_meet_11',
            'p4_meet_12',
            'p4_meet_2',
            'p4_staticsyn_11',
            'p4_staticsyn_13',
            'p4_table_11',
            'p4_table_12'
        ]
