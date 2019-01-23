from os.path import isfile, isdir, join
from os import listdir, makedirs
from pak.util import download, unzip
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D


def plot(ax, human, plot_jid=False, do_scatter=True,
         lcolor="#3498db", mcolor='gray', rcolor="#e74c3c",
         alpha=0.5):
    n_joints, n_channels = human.shape
    assert n_channels == 3

    connect = [
        (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
        (1, 6), (11, 0), (0, 1), (0, 6),
        (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
        (17, 14), (14, 24),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30)
    ]

    LMR = [1, 0, 0, 0, 0,
           0, 2, 2, 2, 2,
           2, 1, 1, 1, 1,
           1, 1, 0, 0, 0,
           0, 0, 0, 0, 2,
           2, 2, 2, 2, 2,
           2, 2, 1, 1, 1]

    for a, b in connect:
        is_middle = (LMR[a] == 1 and LMR[b] == 1) or \
                    (LMR[a] == 0 and LMR[b] == 2) or \
                    (LMR[a] == 2 and LMR[b] == 0)
        color = mcolor
        if not is_middle:
            is_left = LMR[a] == 0 or LMR[b] == 0
            if is_left:
                color = lcolor
            else:
                color = rcolor

        A = human[a]
        B = human[b]

        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]],
                color=color, alpha=alpha)

        if do_scatter:
            ax.scatter(human[:, 0], human[:, 1], human[:, 2],
                       color='gray', alpha=alpha, s=5)

        if plot_jid:
            for i, (x, y, z) in enumerate(human):
                ax.text(x, y, z, str(i))


class CMU_MoCap:

    def __init__(self, data_root, z_is_up=True, store_binary=False):
        """
        :param data_root: root location for data
        :param z_is_up: if True ensure that z points upwards
        :param store_binary: if True store the extracted video sequence as
            numpy binary for faster access
        """
        assert isdir(data_root)

        root = join(data_root, 'cmu_mocap')
        if not isdir(root):
            makedirs(root)

        subject_folder = join(root, 'all_asfamc/subjects')
        if not isdir(subject_folder):
            print("[CMU MoCap] download file")

            zip_files = [
                'allasfamc.zip'
            ]

            for zip_name in zip_files:
                url = 'http://mocap.cs.cmu.edu/' + zip_name
                zip_file = join(root, zip_name)
                if not isfile(zip_file):
                    print('\t[downloading] ', url)
                    download.download(url, zip_file)
                print('\t[unzipping] ', zip_file)
                unzip.unzip(zip_file, root)

        self.subjects = sorted(listdir(subject_folder))
        self.subject_folder = subject_folder
        self.z_is_up = z_is_up
        self.store_binary = store_binary

    def get_actions(self, subject):
        """ returns all action ids for a subject
        :param subject:
        :return:
        """
        subject_loc = join(self.subject_folder, subject)
        a = len(subject) + 1
        amc_files = [f[a:a+2] for f in listdir(subject_loc) if f.endswith('.amc')]
        return sorted(amc_files)

    def get(self, subject, action):
        """
        :param subject:
        :param action:
        :return:
        """
        store_binary = self.store_binary
        subject_loc = join(self.subject_folder, subject)
        assert isdir(subject_loc), subject_loc

        if store_binary:
            npy_file = join(subject_loc, subject + '_' + action + '.npy')
            if isfile(npy_file):
                points3d = np.load(npy_file)
                return points3d

        asf_file = join(subject_loc, subject + '.asf')
        amc_file = join(subject_loc, subject + '_' + action + '.amc')
        assert isfile(asf_file), asf_file
        assert isfile(amc_file), amc_file

        joints = parse_asf(asf_file)
        motions = parse_amc(amc_file)

        n_joints = 31
        n_frames = len(motions)

        points3d = np.empty((n_frames, n_joints, 3), np.float32)

        for frame, motion in enumerate(motions):
            joints['root'].set_motion(motion)
            for jid, j in enumerate(joints.values()):
                points3d[frame, jid] = np.squeeze(j.coordinate)

        if self.z_is_up:
            R = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
            points3d = points3d @ R

        if store_binary:
            np.save(npy_file, points3d)  # file must not exist

        return points3d


# =====================================================================
# External code taken from: Yuxiao Zhou (https://calciferzh.github.io/)
# https://github.com/CalciferZh/AMCParser
# =====================================================================
class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.
    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.
    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.
    length: Length of the bone.
    axis: Axis of rotation for the bone.
    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.
    limits: Limits on each of the channels in the dof specification
    """
    self.name = name
    self.direction = np.reshape(direction, [3, 1])
    self.length = length
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
      rotation = np.deg2rad(motion['root'][3:])
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
    else:
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
      self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
    for child in self.children:
      child.set_motion(motion)

  def draw(self):
    joints = self.to_dict()
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      xs.append(joint.coordinate[0, 0])
      ys.append(joint.coordinate[1, 0])
      zs.append(joint.coordinate[2, 0])
    plt.plot(zs, xs, ys, 'b.')

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        plt.plot(zs, xs, ys, 'r')
    plt.show()

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)
  return frames
