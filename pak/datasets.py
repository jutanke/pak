# Find and download typical datasets for deep learning
#
import zipfile
import urllib.request
import shutil
from os import makedirs
from os.path import join, isfile, exists
from pak import utils


class Dataset:
    """ Dataset base class
    """

    def __init__(self, name, root, verbose=True):
        self.name = name
        self.root = root
        self.verbose = verbose
        if not exists(root):
            makedirs(root)


    def download_and_unzip(self, url):
        """ Downloads and unzips a zipped data file

        """
        dest = join(self.root, self.name)
        if not exists(dest):
            utils.talk("could not find folder " + dest + "...", self.verbose)
            fzip = join(self.root, self.name + ".zip")

            if not isfile(fzip):
                utils.talk("could not find file " + fzip, self.verbose)
                with urllib.request.urlopen(url) as res, open(fzip, 'wb') as f:
                    shutil.copyfileobj(res, f)
                zip_ref = zipfile.ZipFile(fzip, 'r')
                zip_ref.extractall(self.root)
                zip_ref.close()


# =========================================
#  MOT16
# =========================================

class MOT16(Dataset):

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, "MOT16", root, verbose)
        url = 'https://motchallenge.net/data/MOT16.zip'
        self.download_and_unzip(url)


def test():
    utils.talk("LOOL", True)
    print('lol')


def get2DMOT2015():
    """ Gets the 2d MOT2015 dataset
    """
    pass
