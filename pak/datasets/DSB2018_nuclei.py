import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from pak.util import unzip
from scipy.ndimage import imread


class Nuclei:
    """ https://datasciencebowl.com/about/

        Find nuclei in images
        The data has to be downloaded by the user
    """

    def __init__(self, root):
        """
        ctor
        :param root: location of the downloaded data
        """
        self.root = root
        for f in Nuclei.get_required_zips():
            fzip = join(root, f)
            assert isfile(fzip), '.zip file ' + fzip + ' must be downloaded by hand'
            real_file = f[0:-4]  # get rid of .zip
            if real_file.endswith('.csv'):
                already_unzipped = isfile(join(root, real_file))
            else:
                already_unzipped = isdir(join(root, real_file))

            if not already_unzipped:
                if real_file.endswith('.csv'):
                    unzip.unzip(fzip, root)
                else:  # unzip into sub-folder
                    unzip.unzip(fzip, join(root, real_file))

    def get_train(self):
        """
        :return: the train dataset
        """
        train_img_loc = join(self.root, 'stage1_train')
        assert isdir(train_img_loc)

        ids = sorted([f for f in listdir(train_img_loc) if isdir(join(train_img_loc, f))])
        Imgs = []
        Masks = []

        for cid in ids:
            loc = join(train_img_loc, cid); assert isdir(loc)
            img_loc = join(join(loc, 'images'), cid + '.png')
            assert isfile(img_loc)
            Imgs.append(imread(img_loc, mode='L'))

            # masks
            mask_loc = join(loc, 'masks'); assert isdir(mask_loc)
            cur_Mask = []
            for fmask in [join(mask_loc,f) for f in listdir(mask_loc) if f.endswith('.png')]:
                assert isfile(fmask)
                cur_Mask.append(imread(fmask, mode='L'))
            Masks.append(np.array(cur_Mask))

        assert len(Imgs) == len(Masks)
        assert len(Imgs) == len(ids)

        return Imgs, Masks, ids

    def get_test(self):
        """
        :return: the test dataset
        """
        test_img_loc = join(self.root, 'stage1_test')
        assert isdir(test_img_loc)

        ids = sorted([f for f in listdir(test_img_loc) if isdir(join(test_img_loc, f))])
        Imgs = []

        for cid in ids:
            loc = join(test_img_loc, cid); assert isdir(loc)
            img_loc = join(join(loc, 'images'), cid + '.png')
            assert isfile(img_loc)
            Imgs.append(imread(img_loc, mode='L'))

        return Imgs, ids

    # --------- static ---------

    @staticmethod
    def get_required_zips():
        return [
            'stage1_test.zip',
            'stage1_train.zip',
            'stage1_train_labels.csv.zip'
        ]