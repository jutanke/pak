import json; from pprint import pprint
Settings = json.load(open('settings.txt'))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
sys.path.insert(0,'../')
from pak.datasets.CMU_MoCap import CMU_MoCap
import pak.datasets.CMU_MoCap as cmu


data = CMU_MoCap(Settings['data_root'],
                 z_is_up=False,
                 store_binary=False)

joints = data.get('01', '01')
human = joints[0]
minv = -20
maxv = 30

joints, motions = data.get_asf_amc('01', '01')
V, joints = cmu.to_rotation_vector_representation(joints, motions)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


for t in range(0, len(V), 10):
    ax.clear()
    ax.set_xlim([minv, maxv])
    ax.set_ylim([minv, maxv])
    ax.set_zlim([minv, maxv])
    cmu.plot_vector(ax, V[t], joints, plot_jid=False)
    plt.pause(1/120)

plt.show()
