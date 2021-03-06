import numpy as np
import urllib.request
import shutil
import zipfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from os.path import join, isfile, exists


def get_filename_from_url(url):
    return url.split('/')[-1]

def bb_to_plt_plot(x, y, w, h):
    """ Converts a bounding box to parameters
        for a plt.plot([..], [..])
        for actual plotting with pyplot
    """
    X = [x, x,   x+w, x+w, x]
    Y = [y, y+h, y+h, y,   y]
    return X, Y


def tl_br_to_plt_plot(t, l, b, r):
    """ converts two points, top-left and bottom-right into
        bounding box plot data
    """
    assert t < b and l < r
    w = r - l; h = b - t
    X = [l, l, l+w, l+w, l]
    Y = [t, t+h, t+h, t, t]
    return X, Y


def extract_eq(data, col, value):
    """ extract all rows from the data table
        where the values of the column = value

        data must be a 2d matrix:
        shape: (rows, columns)

            col1 col2 col3 ...
        row1
        row2
        row3
        ...

    """
    mask = ((data[:,col] == value) * 1).nonzero()[0]
    return data[mask,:]


def talk(text, verbose):
    """ helper function for printing debug messages
    """
    if verbose:
        print(str(text))


def plot(mats, cols=5, cmap=plt.get_cmap('gray'), size=16):
    """ plot the images in a given list

        X: list of images
    """
    SUBSTITUTE = np.zeros_like(mats[0])
    rows = []
    currentRow = []
    cols = float(cols)
    def add_to_rows():
        while len(currentRow) < cols:
            currentRow.append(SUBSTITUTE)
        rows.append(np.hstack(currentRow))

    for i in range(0, len(mats)):
        M = mats[i]
        minv = np.min(M)
        maxv = np.max(M)
        #if minv < 0 or minv > 255 or maxv < 0 or maxv > 255:
        #    M = translate(M, minv, maxv, 0, 255)

        if i%cols == 0:
            if len(currentRow) > 0:
                add_to_rows()
            currentRow = []
        currentRow.append(M)

    if len(currentRow) > 0:
        add_to_rows()

    I = np.vstack(rows)
    f, ax = plt.subplots(ncols=1)
    f.set_size_inches(size, size)
    ax.imshow(I, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
