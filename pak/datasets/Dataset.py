import numpy as np
# import zipfile
# import tarfile
import urllib.request
import shutil
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.io import loadmat
from skimage.transform import resize
from pak import utils
from pak.util import mpii_human_pose as mpii_hp
from pak.util import unzip
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

    def download_file(self, url, file_name, dest_folder=None):
        """ Only download the file
        """
        if dest_folder is None:
            dest = join(self.root_export, self.name)
        else:
            dest = join(self.root_export, dest_folder)

        if not exists(dest):
            makedirs(dest)

        fname = join(dest, file_name)
        if not isfile(fname):
            utils.talk("Could not find " + fname + " ..., downloading", self.verbose)
            with urllib.request.urlopen(url) as res, open(fname, 'wb') as f:
                utils.talk(url + " downloaded..", self.verbose)
                shutil.copyfileobj(res, f)

        else:  # file exists
            utils.talk("File " + fname + " found :)", self.verbose)




    def download_and_unzip(self, url, zipfile_name=None, dest_folder=None,
        dest_force=True, root_folder=None):
        """ Downloads and unzips a zipped data file

        """
        if dest_folder is None:
            dest = join(self.root, self.name)
        else:
            dest = join(self.root, dest_folder)

        if dest_force:
            export_folder = dest
        else:
            self.root_export

        if root_folder is None:
            root = self.root
        else:
            root = join(self.root, root_folder)
            if not exists(root):
                makedirs(root)

        if not exists(dest):
            utils.talk("could not find folder " + dest + "...", self.verbose)
            if zipfile_name is None:
                fzip = join(root, self.name + ".zip")
            else:
                fzip = join(root, zipfile_name)

            if isfile(fzip):
                utils.talk('found ' + fzip, self.verbose)
            else:
                utils.talk("could not find file " + fzip, self.verbose)
                utils.talk("download from " + url, self.verbose)
                with urllib.request.urlopen(url) as res, open(fzip, 'wb') as f:
                    utils.talk(url + " downloaded..", self.verbose)
                    shutil.copyfileobj(res, f)

            unzip.unzip(fzip, export_folder, self.verbose)
            # if fzip.endswith('.zip'):
            #     utils.talk("unzip " + fzip + " -> " + root, self.verbose)
            #     zip_ref = zipfile.ZipFile(fzip, 'r')
            #     zip_ref.extractall(export_folder)
            #     zip_ref.close()
            # elif fzip.endswith('tar.gz'):
            #     utils.talk("untar " + fzip + " -> " + root, self.verbose)
            #     tar = tarfile.open(fzip, 'r:gz')
            #     tar.extractall(export_folder)
            #     tar.close()
        else:
            utils.talk(dest + ' found :)', self.verbose)
