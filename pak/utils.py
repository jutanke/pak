import urllib.request
import shutil
import zipfile
from os.path import join, isfile, exists


def talk(text, verbose):
    """ helper function for printing debug messages
    """
    if verbose:
        print(str(text))

