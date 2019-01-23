import json; from pprint import pprint
Settings = json.load(open('settings.txt'))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
sys.path.insert(0,'../')
from pak.datasets.CMU_MoCap import CMU_MoCap, plot


data = CMU_MoCap(Settings['data_root'])

joints = data.get('01', '01')
print("joints:", joints.shape)

FRAME = 0

minv = 1.1 * np.min(joints)
maxv = 1.1 * np.max(joints)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


for t in range(0, 430, 10):
    ax.clear()
    ax.set_xlim([minv, maxv])
    ax.set_ylim([minv, maxv])
    ax.set_zlim([minv, maxv])
    plot(ax, joints[t], plot_jid=False)
    plt.pause(1/120)

plt.show()
