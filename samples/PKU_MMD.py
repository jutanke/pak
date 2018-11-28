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

skel1, _, labels1 = data.get_3d('0002-L')
skel2, _, labels2 = data.get_3d('0002-R')

fig = plt.figure(figsize=(16, 8))
R = 1
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')


def plot(ax, skel):
    s1 = skel[t]
    ax.clear()
    ax.set_xlim(-R, R)
    ax.set_xlabel('x')
    ax.set_ylim(-2 - R * 2, 0)
    ax.set_ylabel('y')
    ax.set_zlim(-R, R)
    ax.set_zlabel('z')
    plot_skeleton(ax, s1)


for t in range(2000):
    label = labels1[t]
    plot(ax1, skel1)
    plot(ax2, skel2)

    txt = 'None' if label == 0 else data.action_id_to_action_name[label]
    ax1.set_title(txt)
    ax2.set_title(txt)

    plt.pause(1/33.33)

plt.show()
