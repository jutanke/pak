import json; from pprint import pprint
Settings = json.load(open('settings.txt'))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
sys.path.insert(0,'../')
from pak.datasets.CMU_MoCap import CMU_MoCap


data = CMU_MoCap(Settings['data_root'])

frames, joints = data.get('01', '01')

pts = joints[0, :, 0:3]

minv = 1.1 * np.min(pts)
maxv = 1.1 * np.max(pts)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for jid, (x, y, z) in enumerate(pts):
    ax.scatter(x, y, z, color='blue')
    ax.text(x, y, z, str(jid))

# for t in range(1000):
#     pts = joints[t, :, 0:3]
#     ax.clear()
#     ax.set_xlim([minv, maxv])
#     ax.set_ylim([minv, maxv])
#     ax.set_zlim([minv, maxv])
#     ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
#     plt.pause(1/120)

plt.show()
