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

#data.train_videos

skel1, skel2 = data.get_3d('0002-L')

s1 = skel1[2000]
s2 = skel2[0]

fig = plt.figure(figsize=(12, 12))
R = 1
ax = fig.add_subplot(111, projection='3d')

for t in range(2000):
    s1 = skel1[t]
    ax.clear()
    ax.set_xlim(-R, R)
    ax.set_xlabel('x')
    ax.set_ylim(-2 - R * 2, 0)
    ax.set_ylabel('y')
    ax.set_zlim(-R, R)
    ax.set_zlabel('z')
    plot_skeleton(ax, s1)
    plt.pause(1/33.33)

plt.show()
