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

from pak.datasets.CAD_120 import CAD_120

data = CAD_120(root)

suj1 = data.get_subject(1)

vid = suj1['arranging_objects']['0510175431']

pts3d = data.get_3d_points_from_skel(vid['skeleton'])

person = pts3d[0]

# =========

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

data.plot(ax, person, plot_jids=True)

plt.axis('equal')
plt.show()
