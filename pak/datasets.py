# Find and download typical datasets for deep learning

import numpy as np
import zipfile
import tarfile
import urllib.request
import shutil
from os import makedirs, listdir
from os.path import join, isfile, exists, splitext
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.io import loadmat
from skimage.transform import resize
from pak import utils
from pak.util import mpii_human_pose as mpii_hp
import h5py
from enum import Enum

class Dataset:
    """ Dataset base class
    """

    def __init__(self, name, root, verbose=True):
        self.name = name
        self.root = root
        self.root_export = root
        self.verbose = verbose
        if not exists(root):
            makedirs(root)

    def unzip(self):
        """ Unzip the name file into the root_export directory
            If the zip file is not found an exception is thrown
        """
        dest = join(self.root, self.name)
        if not exists(dest):
            fzip = join(self.root, self.name + ".zip")
            if not isfile(fzip):
                utils.talk('Could not find ' + fzip, self.verbose)
                raise Exception('Could not find ' + fzip)

            zip_ref = zipfile.ZipFile(fzip, 'r')
            utils.talk("unzip " + fzip + " -> " + self.root, self.verbose)
            zip_ref.extractall(self.root_export)
            zip_ref.close()
        else:
            utils.talk(dest + ' found :)', self.verbose)

    def download_and_unzip(self, url, zipfile_name=None, dest_folder=None):
        """ Downloads and unzips a zipped data file

        """
        if dest_folder is None:
            dest = join(self.root, self.name)
        else:
            dest = join(self.root, dest_folder)

        if not exists(dest):
            utils.talk("could not find folder " + dest + "...", self.verbose)
            if zipfile_name is None:
                fzip = join(self.root, self.name + ".zip")
            else:
                fzip = join(self.root, zipfile_name)

            if isfile(fzip):
                utils.talk('found ' + fzip, self.verbose)
            else:
                utils.talk("could not find file " + fzip, self.verbose)
                utils.talk("download from " + url, self.verbose)
                with urllib.request.urlopen(url) as res, open(fzip, 'wb') as f:
                    utils.talk(url + " downloaded..", self.verbose)
                    shutil.copyfileobj(res, f)

            if fzip.endswith('.zip'):
                utils.talk("unzip " + fzip + " -> " + self.root, self.verbose)
                zip_ref = zipfile.ZipFile(fzip, 'r')
                zip_ref.extractall(self.root_export)
                zip_ref.close()
            elif fzip.endswith('tar.gz'):
                utils.talk("untar " + fzip + " -> " + self.root, self.verbose)
                tar = tarfile.open(fzip, 'r:gz')
                tar.extractall(self.root_export)
                tar.close()
        else:
            utils.talk(dest + ' found :)', self.verbose)


# =========================================
#  MOTXX
# =========================================

class MOT_X(Dataset):

    def __init__(self, root, root_export, name, url, verbose=True, resize=None):
        Dataset.__init__(self, name, root, verbose)
        self.root_export = root_export
        self.download_and_unzip(url)
        self.resize = resize


    def get_raw(self, folder, parent):
        """ get the raw train data
        """
        root = join(self.root_export, parent)
        loc = join(root, folder)

        # X
        img_loc = join(loc, "img1")
        imgs = sorted([join(img_loc, f) \
                for f in listdir(img_loc) if isfile(join(img_loc, f))])
        if self.resize is None:
            X = np.array([imread(f) for f in imgs], 'uint8')
        else:
            X = np.array([imresize(imread(f), size=self.resize) for f in imgs], 'uint8')
        utils.talk(self.name + ' X loaded', self.verbose)

        # Y-det
        det_txt = join(join(loc, "det"), 'det.txt')
        Y_det = np.loadtxt(det_txt, delimiter=',')
        utils.talk(self.name + ' Y_det loaded', self.verbose)

        return X, Y_det

    def get_test(self, folder):
        """ Gets the raw MOT data for testing
        """
        parent = 'test'
        return MOT_X.get_raw(self, folder, parent=parent)

    def get_train(self, folder):
        """ Gets the raw MOT data for training
        """
        parent = 'train'
        X, Y_det = MOT_X.get_raw(self, folder, parent=parent)

        # Y-gt
        root = join(self.root_export, parent)
        loc = join(root, folder)
        gt_txt = join(join(loc, "gt"), 'gt.txt')
        Y_gt = np.loadtxt(gt_txt, delimiter=',')
        utils.talk(self.name + ' Y_gt loaded', self.verbose)

        return X, Y_det, Y_gt


    def get_train_folders(self):
        raise NotImplementedError("Must be overriden")


# =========================================
#  MOT16
# =========================================

class MOT16(MOT_X):
    """ MOT16 dataset
    """

    def __init__(self, root, verbose=True, resize=None):
        root_export = join(root, "MOT16")  # force dir name for unzipping
        MOT_X.__init__(self, root, root_export, "MOT16",
                "https://motchallenge.net/data/MOT16.zip", verbose, resize)


    def get_train_folders(self):
        return ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", \
            "MOT16-10", "MOT16-11", "MOT16-13"]

    def get_test_folders(self):
        return ["MOT16-01", "MOT16-03", "MOT16-06", "MOT16-07", \
            "MOT16-08", "MOT16-12", "MOT16-14"]

    def label_id_to_class(self, label_id):
        """ converts the label number id to
            the true label content
        """
        return ["NONE",
                "Pedestrian",
                "Person on vehicle",
                "Car",
                "Bycicle",
                "Motorbike",
                "Non motorized vehicle",
                "Static person",
                "Distractor",
                "Occluder",
                "Occluder on the ground",
                "Occluder full",
                "Reflection" ][int(label_id)]

# =========================================
#  MOT15
# =========================================

class MOT152D(MOT_X):
    """ MOT15 2d dataset
    """

    def __init__(self, root, verbose=True, resize=None):
        MOT_X.__init__(self, root, root, "2DMOT2015",
                "https://motchallenge.net/data/2DMOT2015.zip", verbose, resize)
        self.root_export = join(root, "2DMOT2015")  # the dirs name after unzip


    def get_train_folders(self):
        return ["ADL-Rundle-6", "ETH-Bahnhof", "ETH-Sunnyday", "KITTI-17", \
            "TUD-Campus", "Venice-2", "ADL-Rundle-8", "ETH-Pedcross2", \
            "KITTI-13", "PETS09-S2L1", "TUD-Stadtmitte"]

    def get_test_folders(self):
        return ["ADL-Rundle-1", "ADL-Rundle-3", "AVG-TownCentre", \
            "ETH-Jelmoli", "KITTI-16", "PETS09-S2L2", "Venice-1", \
            "ETH-Crossing", "ETH-Linthescher", "KITTI-19", "TUD-Crossing"]

# =========================================
#  MARKET 1501
# =========================================

class Market1501(Dataset):
    """ Market1501 dataset
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, "Market-1501-v15.09.15", root, verbose)
        self.download_and_unzip('NOT-YET-THERE')
        self.root_export = join(root, "Market-1501-v15.09.15")

    def get_train(self):
        return self.get_raw('bounding_box_train')

    def get_test(self):
        return self.get_raw('bounding_box_test')

    def get_raw(self, folder):
        """ gets the raw identity pictures
        """
        loc = join(self.root_export, folder)
        imgs = sorted([f \
                    for f in listdir(loc) if isfile(join(loc, f)) and \
                      f.endswith('jpg')])
        X = np.array([imread(join(loc, f)) for f in imgs], 'uint8')

        identities = np.array([int(f[0:2]) if f.startswith('-1') else int(f[0:4]) \
            for f in imgs])
        cameras = np.array([int(f[4]) if f.startswith('-1') else int(f[6]) \
            for f in imgs])
        sequences = np.array([int(f[6]) if f.startswith('-1') else int(f[8]) \
            for f in imgs])
        frames = np.array([int(f[8:14]) if f.startswith('-1') else int(f[10:16]) \
            for f in imgs])

        Y = np.vstack((identities, cameras, sequences, frames))

        return X, np.rollaxis(Y, 1)


# =========================================
#  MPII-Human-Pose
# =========================================
class MPII_human_pose(Dataset):
    """ MPII Human Pose dataset
    """

    def __init__(self, root, verbose=True):
        #mpii_hp.test()
        Dataset.__init__(self, 'mpii_human_pose_v1', root, verbose)

        url_data = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz'
        url_anno = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip'
        self.download_and_unzip(url_anno,
            zipfile_name='mpii_human_pose_v1_u12_2.zip',
            dest_folder='mpii_human_pose_v1_u12_2')
        self.root_export = join(root, 'mpii_human_pose_v1_u12_2')
        self.download_and_unzip(url_data,
            zipfile_name='mpii_human_pose_v1.tar.gz',
            dest_folder=join('mpii_human_pose_v1_u12_2', 'images'))


    def get_annotation(self):
        """ reads the annotation and returns it
        """
        mat = join(self.root_export, "mpii_human_pose_v1_u12_1.mat")
        M = loadmat(mat)
        M = M['RELEASE']
        AL = M['annolist'][0][0][0]
        TR = M['img_train'][0][0][0]


        n = len(AL)
        n = 10
        result = []
        for i in range(n):
            print("stuff", i)
            e = AL[i]   # get the image meta data, nbr of persons,
                        # person joints, etc.
            print(e)
            is_training_data = TR[i] == 1
            data = mpii_hp.get_data(e, is_training_data)
            result.append(data)
        return result


# =========================================
#  LSPE
# =========================================
class LSPE(Dataset):
    """ Leeds Sports Pose Extended
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, 'lspe', root, verbose)
        url = 'http://sam.johnson.io/research/lspet_dataset.zip'
        self.root_export = join(root, "lspe")
        self.download_and_unzip(url)

    def get_raw(self):
        """ gets the raw data without any resizing.
            Attention: Image sizes vary!
        """
        image_folder = join(self.root_export, 'images')
        joint_mat = join(self.root_export, 'joints.mat')

        Y = np.rollaxis(loadmat(joint_mat)['joints'], 2, 0)
        n, _, _ = Y.shape

        imgs = sorted([f for f in listdir(image_folder) if \
                        isfile(join(image_folder, f))])
        X = np.array([imread(join(image_folder, f)) for f in imgs])

        return X, Y

# =========================================
#  DukeMTMC-reID
# =========================================

class DukeMTMC_reID(Dataset):
    """ DukeMTMC-reID dataset, very similar to Market1501
    """

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, "DukeMTMC-reID", root, verbose)
        Dataset.unzip(self)
        self.root_export = join(root, "DukeMTMC-reID")

    def get_train(self):
        return self.get_raw('bounding_box_train')

    def get_test(self):
        return self.get_raw('bounding_box_test')

    def get_raw(self, folder):
        """ gets the raw identity pictures
        """
        loc = join(self.root_export, folder)
        imgs = sorted([f \
                    for f in listdir(loc) if isfile(join(loc, f)) and \
                      f.endswith('jpg')])
        X = np.array([imread(join(loc, f)) for f in imgs])

        identities = np.array([int(f[0:4]) for f in imgs])
        cameras = np.array([int(f[6]) for f in imgs])
        frames = np.array([int(f[9:15]) for f in imgs])

        Y = np.vstack((identities, cameras, frames))

        return X, np.rollaxis(Y, 1)

# =========================================
#  Hand
# =========================================
class Hand_config(Enum):
    Default = 1
    Square = 2
    AABB = 3

class Hand(Dataset):
    """
    hand dataset (http://www.robots.ox.ac.uk/~vgg/data/hands/)
    """

    def __init__(self, root, verbose=True):
        """ create a hand dataset
        """
        Dataset.__init__(self, "hand_dataset", root, verbose)
        url = 'http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz'
        #self.root_export = join(root, "lspe")
        self.download_and_unzip(url,
            zipfile_name='hand_dataset.tar.gz')
        self.root_export = join(root, "hand_dataset")

    def get_test(self, config=Hand_config.Default):
        return self.get_raw(join('test_dataset', 'test_data'), config)

    def get_train(self, config=Hand_config.Default):
        return self.get_raw(join('training_dataset', 'training_data'),\
            config)

    def get_val(self, config=Hand_config.Default):
        return self.get_raw(join('validation_dataset', 'validation_data'),\
            config)

    def get_raw(self, subfolder, config):
        """ test vs train vs validation

            make_square: if True a 4th point is added to enclose the
                hand
        """
        make_square = config is Hand_config.Square or config is Hand_config.AABB
        aabb = config is Hand_config.AABB

        path = join(self.root_export, subfolder)

        # # annotations
        path_anno = join(path, 'annotations')
        path_imgs = join(path, 'images')

        slates = sorted([splitext(f)[0] for f in listdir(path_imgs) if \
            isfile(join(path_anno, splitext(f)[0] + '.mat')) and \
            isfile(join(path_imgs, splitext(f)[0] + '.jpg'))])

        X = []
        Y = []

        for f in slates:
            img_file = join(path_imgs, f + '.jpg')
            ann_file = join(path_anno, f + '.mat')

            x = imread(img_file)
            X.append(x)

            # --- y ---
            M = loadmat(ann_file)['boxes'][0]
            nbr_boxes = len(M)
            Frame = []
            for i in range(nbr_boxes):
                single_hand = M[i][0][0]
                Hand = []
                for j in range(3):
                    e = single_hand[j][0]
                    Hand.append(np.array((e[1], e[0])))  # first X, then Y ...
                if make_square:
                    v1 = Hand[0]
                    v2 = Hand[1]
                    v3 = Hand[2]

                    # we want to determine {4} given (1),(2),(3)
                    # (1)--(w)--(2)
                    #  .         |
                    #  .        (h)
                    #  .         |
                    # {4}. . . .(3)
                    direction = v3 - v2
                    v4 = v1 + direction
                    Hand.append(v4)

                    if aabb:
                        x1,y1 = v1
                        x2,y2 = v2
                        x3,y3 = v3
                        x4,y4 = v4

                        Hand = [
                            np.array([max([x1,x2,x3,x4]), max([y1,y2,y3,y4])]),
                            np.array([min([x1,x2,x3,x4]), min([y1,y2,y3,y4])])
                        ]


                Frame.append(Hand)
            Y.append(Frame)

        return np.array(X), Y

# =========================================
#  CUHK03
# =========================================

class cuhk03(Dataset):
    """ cuhk03 dataset
    """

    def __init__(self, root, verbose=True, target_w=100, target_h=256):
        """ create cuhk03 dataset

            root: data folder that stores all files
            verbose: boolean flag to tell if the class should talk in debug mode
                or not
            target_w: resize all images to fit target_w
            target_h: resize all images to fit target_h
        """
        Dataset.__init__(self, "cuhk03_release", root, verbose)
        Dataset.unzip(self)
        self.hd5file = join(join(root, self.name), 'cuhk-03.mat')
        self.target_w = target_w
        self.target_h = target_h


    def get_detected(self):
        """ gets the images that were detected by a automated person detector
        """
        return self.get_raw('detected')

    def get_labeled(self):
        """ gets the images that were humanly-annotated
        """
        return self.get_raw('labeled')

    def get_raw(self, folder):
        f = h5py.File(self.hd5file,'r+')
        data = f[folder]
        _, n = data.shape
        tw = self.target_w
        th = self.target_h

        current_id = 1

        Imgs = []
        Ids = []

        for view in range(n):
            utils.talk("cuhk03: view \"" + folder + "\" " + \
                str(view+1) + "/" + str(n), self.verbose)
            V = f[data[0][view]]
            ten, M = V.shape  # M is number of identities
            for i in range(M):
                for j in range(ten):
                    img = f[V[j][i]].value
                    if len(img.shape) == 3:
                        img = np.swapaxes(img, 0, 2)
                        img = resize(img, (th, tw), mode='constant')
                        Imgs.append((img * 255).astype('uint8'))
                        Ids.append(current_id)
                current_id += 1

        X = np.array(Imgs)
        Y = np.array(Ids, 'int32')
        return X, Y
