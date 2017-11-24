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

def get_data(image_meta):
    """ the meta data has a horrible shape..
        I guess that's because of the extraction
        from matlab code..
    """

    result = ImageMetaData()

    result.name= image_meta[0][0][0][0][0]
    _, nbr_persons = image_meta[1].shape
    result.vidx = image_meta[2][0][0]
    result.frame_sec = image_meta[3][0][0]

    for person in range(nbr_persons):
        p = Person()
        x1 = image_meta[1][0][person][0][0][0]
        y1 = image_meta[1][0][person][1][0][0]
        x2 = image_meta[1][0][person][2][0][0]
        y2 = image_meta[1][0][person][3][0][0]

        p.head_bb = ((x1, y1), (x2, y2))

        joints = image_meta[1][0][person][4][0][0][0][0]
        p.joints = compress_joints(joints)
        result.people.append(p)

    return result
