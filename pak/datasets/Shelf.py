from os import makedirs
from os.path import join, isfile, isdir
from pak.util import download
from pak.util import unzip


class Shelf:
    """
    Shelf dataset for multiple human multi-view pose estimation
        http://campar.in.tum.de/Chair/MultiHumanPose
    """

    def __init__(self, root, verbose=True):
        """

        :param root:
        :param verbose:
        """
        if verbose:
            print('\n**Shelf dataset**')
        data_root = join(root, 'tum_shelf')
        if not isdir(data_root):
            makedirs(data_root)

        self.verbose = verbose
        self.data_root = data_root

        # download data
        url = 'http://campar.cs.tum.edu/files/belagian/multihuman/Shelf.tar.bz2'
        data_folder = join(data_root, 'Shelf')
        if not isdir(data_folder):
            zip_filename = join(data_root, 'Shelf.tar.bz2')
            if not isfile(zip_filename):
                if verbose:
                    print('\tdownload ' + url)
                download.download(url, zip_filename)

            if verbose:
                print('\nunzip ' + zip_filename)
                unzip.unzip(zip_filename, data_root, verbose)
                
        if verbose:
            print('\n')
