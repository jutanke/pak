import urllib.request
import shutil
import zipfile
from os.path import join, isfile, exists

def talk(text, verbose):
    """ helper function for printing debug messages
    """
    if verbose:
        print(str(text))

def download_and_unzip(dest, fzip, url, verbose=False):
    """ Downloads and unzips a zipped data file

    dest: {String}, folder that contains the unzipped data
    fzip: {String}, file name of the zipped downloaded file
    url: {String}, URL for the zip file
    """
    if exists(dest):
        talk(dest + " exists, returning..", verbose)


