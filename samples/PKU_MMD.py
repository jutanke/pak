import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json; from pprint import pprint
Settings = json.load(open('settings.txt'))
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'../')
from os.path import isdir

root = Settings['data_root']

from pak.datasets.PKU_MMD import PKU_MMD, plot_skeleton

data = PKU_MMD(root)

skel1, skel2, labels = data.get_3d('0075-M')

fig = plt.figure(figsize=(12, 12))
R = 1
ax = fig.add_subplot(111, projection='3d')


def plot(ax, skel, t):
    s1 = skel[t]
    ax.set_xlim(-R, R)
    ax.set_xlabel('x')
    ax.set_ylim(-2 - R * 2, 0)
    ax.set_ylabel('y')
    ax.set_zlim(-R, R)
    ax.set_zlabel('z')
    plot_skeleton(ax, s1)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


for t in range(2000):
    label = labels[t]
    ax.clear()

    txt = 'None' if label == 0 else data.action_id_to_action_name[label]
    plot(ax, skel1, t)

    if not isclose(np.mean(np.abs(skel2)), 0):
        plot(ax, skel2, t)

    ax.set_title(txt)
    plt.pause(1/66.66)

plt.show()
