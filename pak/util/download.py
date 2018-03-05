import numpy as np
import urllib.request
import shutil
import subprocess
import time
import shutil
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext


def os_has_wget():
    """
    Checks if the OS has wget
    :return: True if wget is available, False o/w
    """
    command = ["which", 'wget']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, err = process.communicate()
    return len(output) > 1  # path to binary OR empty


def download(url, fname):
    """ download file from url
    :param url:
    :param fname: file location for the downloaded file
    :return:
    """
    with urllib.request.urlopen(url) as res, open(fname, 'wb') as f:
        shutil.copyfileobj(res, f)


def download_with_login(url, dir, user, password):
    """ download file from url where a http login is
        required
    :param url:
    :param dir: file location to where to put the downloaded file
    :param user:  username
    :param password: and password
    :return:
    """
    assert os_has_wget(), "we need an OS that has 'wget'"
    command = ['wget',
               '--http-user=' + user,
               '--http-passwd=' + password,
               '-i',
               url]
    if not isdir(dir):
        makedirs(dir)
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    process.wait()
    time.sleep(0.5)
    fname = url.split("/")[-1]
    moved_fname = join(dir, fname)
    shutil.move(fname, moved_fname)
    time.sleep(0.5)
    return process.communicate()