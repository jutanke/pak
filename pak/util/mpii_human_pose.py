import numpy as np

class ImageMetaData:
    vidx = -1
    frame_sec = -1
    name = None
    people = []

    def __str__(self):
        return "vidx:" + str(self.vidx) + \
            ", frame_sec:" + str(self.frame_sec) + \
            ", name:" + self.name

class Person:
    head_bb = None
    joints = None


def get_all_shapes():
    return [
        ("f_720_1280.npy", (11971, 720, 1280, 3), '(720, 1280, 3)'),
        ("f_1080_1920.npy", (6864, 1080, 1920, 3), '(1080, 1920, 3)'),
        ("f_480_854.npy", (1092, 480, 854, 3), '(480, 854, 3)'),
        ("f_468_854.npy", (17, 468, 854, 3), '(468, 854, 3)'),
        ("f_534_1280.npy", (18, 534, 1280, 3), '(534, 1280, 3)'),
        ("f_480_640.npy", (2680, 480, 640, 3), '(480, 640, 3)'),
        ("f_480_600.npy", (134, 480, 600, 3), '(480, 600, 3)'),
        ("f_480_848.npy", (118, 480, 848, 3), '(480, 848, 3)'),
        ("f_720_960.npy", (281, 720, 960, 3), '(720, 960, 3)'),
        ("f_544_1280.npy", (5, 544, 1280, 3), '(544, 1280, 3)'),
        ("f_480_786.npy", (6, 480, 786, 3), '(480, 786, 3)'),
        ("f_1080_1440.npy", (234, 1080, 1440, 3), '(1080, 1440, 3)'),
        ("f_480_648.npy", (17, 480, 648, 3), '(480, 648, 3)'),
        ("f_470_854.npy", (150, 470, 854, 3), '(470, 854, 3)'),
        ("f_480_646.npy", (39, 480, 646, 3), '(480, 646, 3)'),
        ("f_480_654.npy", (240, 480, 654, 3), '(480, 654, 3)'),
        ("f_480_720.npy", (458, 480, 720, 3), '(480, 720, 3)'),
        ("f_480_624.npy", (11, 480, 624, 3), '(480, 624, 3)'),
        ("f_720_1270.npy", (8, 720, 1270, 3), '(720, 1270, 3)'),
        ("f_714_1280.npy", (11, 714, 1280, 3), '(714, 1280, 3)'),
        ("f_480_800.npy", (36, 480, 800, 3), '(480, 800, 3)'),
        ("f_480_638.npy", (33, 480, 638, 3), '(480, 638, 3)'),
        ("f_480_852.npy", (21, 480, 852, 3), '(480, 852, 3)'),
        ("f_480_656.npy", (63, 480, 656, 3), '(480, 656, 3)'),
        ("f_478_854.npy", (22, 478, 854, 3), '(478, 854, 3)'),
        ("f_474_854.npy", (72, 474, 854, 3), '(474, 854, 3)'),
        ("f_720_406.npy", (8, 720, 406, 3), '(720, 406, 3)'),
        ("f_720_1272.npy", (103, 720, 1272, 3), '(720, 1272, 3)'),
        ("f_1080_1906.npy", (53, 1080, 1906, 3), '(1080, 1906, 3)'),
        ("f_480_712.npy", (26, 480, 712, 3), '(480, 712, 3)'),
        ("f_716_1280.npy", (2, 716, 1280, 3), '(716, 1280, 3)'),
        ("f_480_630.npy", (4, 480, 630, 3), '(480, 630, 3)'),
        ("f_1072_1920.npy", (1, 1072, 1920, 3), '(1072, 1920, 3)'),
        ("f_480_576.npy", (11, 480, 576, 3), '(480, 576, 3)'),
        ("f_1080_1916.npy", (21, 1080, 1916, 3), '(1080, 1916, 3)'),
        ("f_476_854.npy", (37, 476, 854, 3), '(476, 854, 3)'),
        ("f_720_974.npy", (3, 720, 974, 3), '(720, 974, 3)'),
        ("f_720_1266.npy", (3, 720, 1266, 3), '(720, 1266, 3)'),
        ("f_720_1278.npy", (14, 720, 1278, 3), '(720, 1278, 3)'),
        ("f_408_720.npy", (8, 408, 720, 3), '(408, 720, 3)'),
        ("f_720_1080.npy", (5, 720, 1080, 3), '(720, 1080, 3)'),
        ("f_480_842.npy", (3, 480, 842, 3), '(480, 842, 3)'),
        ("f_480_632.npy", (13, 480, 632, 3), '(480, 632, 3)'),
        ("f_640_1280.npy", (13, 640, 1280, 3), '(640, 1280, 3)'),
        ("f_988_1920.npy", (2, 988, 1920, 3), '(988, 1920, 3)'),
        ("f_720_900.npy", (3, 720, 900, 3), '(720, 900, 3)'),
        ("f_720_1200.npy", (3, 720, 1200, 3), '(720, 1200, 3)'),
        ("f_480_840.npy", (7, 480, 840, 3), '(480, 840, 3)'),
        ("f_432_768.npy", (4, 432, 768, 3), '(432, 768, 3)'),
        ("f_684_1280.npy", (6, 684, 1280, 3), '(684, 1280, 3)'),
        ("f_480_810.npy", (6, 480, 810, 3), '(480, 810, 3)'),
        ("f_480_360.npy", (8, 480, 360, 3), '(480, 360, 3)'),
        ("f_480_272.npy", (7, 480, 272, 3), '(480, 272, 3)'),
        ("f_720_480.npy", (1, 720, 480, 3), '(720, 480, 3)'),
        ("f_960_1920.npy", (1, 960, 1920, 3), '(960, 1920, 3)'),
        ("f_480_806.npy", (2, 480, 806, 3), '(480, 806, 3)'),
        ("f_400_854.npy", (2, 400, 854, 3), '(400, 854, 3)'),
        ("f_720_948.npy", (2, 720, 948, 3), '(720, 948, 3)'),
        ("f_720_1002.npy", (1, 720, 1002, 3), '(720, 1002, 3)')
    ]

def nbr_to_joint(i):
    if i == 0:
        return "r ankle"
    elif i == 1:
        return "r knee"
    elif i == 2:
        return "r hip"
    elif i == 3:
        return "l hip"
    elif i == 4:
        return "l knee"
    elif i == 5:
        return "l ankle"
    elif i == 6:
        return "pelvis"
    elif i == 7:
        return "thorax"
    elif i == 8:
        return "upper neck"
    elif i == 9:
        return "head top"
    elif i == 10:
        return "r wrist"
    elif i == 11:
        return "r elbow"
    elif i == 12:
        return "r shoulder"
    elif i == 13:
        return "l shoulder"
    elif i == 14:
        return "l elbow"
    elif i == 15:
        return "l wrist"
    else:
        raise Exception("Invalid joint id:" + str(i))

def test_if_joints_are_first(joints):
    for F, _, _, _ in joints:
        if F > 15:
            return False
        else:
            return True

def compress_joints(joints):
    obj = []
    if not test_if_joints_are_first(joints):
        for x, y,joint_type, visible in joints:
            joint_type = joint_type[0][0]
            x = x[0][0]
            y = y[0][0]
            visible = int(visible[0]) if len(visible) == 1 else -1
            obj.append((joint_type, x, y, visible))
        return obj
    else:
        for joint_type, x, y, visible in joints:
            joint_type = joint_type[0][0]
            x = x[0][0]
            y = y[0][0]
            visible = int(visible[0]) if len(visible) == 1 else -1
            obj.append((joint_type, x, y, visible))
        return obj

def get_data(image_meta, is_training_data):
    """ the meta data has a horrible shape..
        I guess that's because of the extraction
        from matlab code..
    """

    result = ImageMetaData()

    result.name= image_meta[0][0][0][0][0]
    _, nbr_persons = image_meta[1].shape
    result.vidx = image_meta[2][0][0] if is_training_data else -1
    result.frame_sec = image_meta[3][0][0] if is_training_data else -1

    for person in range(nbr_persons):
        p = Person()
        if is_training_data:
            x1 = image_meta[1][0][person][0][0][0]
            y1 = image_meta[1][0][person][1][0][0]
            x2 = image_meta[1][0][person][2][0][0]
            y2 = image_meta[1][0][person][3][0][0]

            p.head_bb = ((x1, y1), (x2, y2))

            joints = image_meta[1][0][person][4][0][0][0][0]
            p.joints = compress_joints(joints)
            result.people.append(p)

    return result
