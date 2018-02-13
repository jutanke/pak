from pak.datasets.Dataset import Dataset
import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
from scipy.ndimage import imread
from scipy.io import loadmat
from pak import utils


class MARCOnI(Dataset):
    """ MPII http://marconi.mpi-inf.mpg.de/#download
    """

    # ---------- STATIC ------------
    @staticmethod
    def get_image_file_extension(name):
        """
        different videos have different file extensions (jpg vs png)
        :param name: name of the video
        :return: the correct image file extensio
        """
        if name in ["Soccer", "Kickbox", "Soccer2", "Volleyball", "Walk1", "Walk2"]:
            return "jpg"
        elif name in ["SBoard"]:
            return "png"
        else:
            raise Exception("Could not find MARCOnI name:" + name)

    # @staticmethod
    # def get_cnn_to_image_factor(name):
    #     """
    #     different videos have different scaling factors for the size of the
    #     CNN detectors in comparison to the video
    #     :param name: of the video
    #     :return: factor: cnn_w = image_w/factor, cnn_h = image_h/factor
    #     """
    #     if name in ["Soccer", "Kickbox"]:
    #         return 2
    #     elif name in ["SBoard"]:
    #         return 4
    #     else:
    #         raise Exception("Could not find MARCOnI name:" + name)

    # ------------------------------

    def __init__(self, root, verbose=True):
        Dataset.__init__(self, 'marconi', root, verbose)
        utils.talk('(MARCOnl)', verbose)

        # This is a horrible hack but it seems that
        # the python tools cannot unzip certain zip files
        # (that seem to be zipped on OSX) and I have no
        # time to find a 'good' solution so the ugly hack
        # is: use the system tools which for some reason
        # work! (on LINUX!!) --> see MARCOnI
        self.force_unzip_with_os_tools = True

        self._verbose = verbose
        self.root_export = join(root, 'marconi')
        self.download_set("Soccer")
        self.download_set("Kickbox")
        self.download_set("SBoard")
        self.download_set("Soccer2")
        self.download_set("Walk1")
        self.download_set("Walk2")
        self.download_set("Volleyball")
        self.download_set("Juggling")
        self.download_set("Run2")

    def get_annotations(self, NAME):
        """ read the annotation matlab file
        """
        root_dir = join(self.root, 'marconi/' + NAME)
        fAnnot = join(root_dir, 'Annotations.mat')

        annolist = loadmat(fAnnot)['annolist'][0]

        num_cams,num_frames,h,w,_ = self.get_video_shape(NAME)

        annotation_per_camera = []
        for _ in range(num_cams):
            annotation_per_camera.append([])

        for elem in annolist:
            name,individuals,_ = elem
            name = name[0][0][0][0]
            current_cam = int(name[-11:-9])
            current_frame = int(name[-8:-4])

            # we need to make sure that we have annotations for each frame
            assert current_frame == len(annotation_per_camera[current_cam])

            annotation_per_individuum = []

            if len(individuals) == 0:  # there are NO detections in this frame
                annotation_per_camera[current_cam].append([])
            else:
                individuals = individuals[0]

                for individual in individuals:
                    if individual is None:
                        annotation_per_individuum.append(None)
                    else:
                        if len(individual) == 9:
                            a,b,c,d,_,_,_,h,_ = individual
                            if len(h[0]) == 0:  # missing data
                                annotation_per_individuum.append(None)
                                continue
                        elif len(individual) == 8:
                            a,b,c,d,e,_,h,_ = individual
                            if len(h) == 0:  # missing data
                                annotation_per_individuum.append(None)
                                continue
                        elif len(individual) == 6 or \
                                len(individual) == 7:  # missing data
                            annotation_per_individuum.append(None)
                            continue
                        else:
                            raise Exception("weird individual annotation:" +
                                            str(current_cam) + "/" + str(current_frame) + \
                                            " len:" + str(len(individual)))

                        a = a[0][0]; b = b[0][0]
                        c = c[0][0]; d = d[0][0]
                        #e = e[0][0]; g = g[0][0]; f = f[0]

                        head_top_left = (min(a, c), min(b, d))
                        head_bottom_right = (max(a, c), max(b, d))

                        Joints = np.zeros((12, 3), 'uint16')
                        joints = h[0][0][0][0]
                        for jidx, joint in enumerate(joints):
                            x, y, pid, visible = joint
                            x = x[0][0];  y = y[0][0];
                            pid = pid[0][0]; visible = visible[0][0]
                            Joints[pid,0] = x
                            Joints[pid,1] = y
                            Joints[pid,2] = visible

                        annotation_per_individuum.append(
                            ((head_top_left, head_bottom_right),
                            Joints)
                        )

                annotation_per_camera[current_cam].append(
                    annotation_per_individuum)

        return annotation_per_camera

    def get_memmapped_file_names(self, name):
        """ get the file names for the memmaped files
        """
        root_dir = join(self.root, 'marconi/' + name)
        fImages = join(root_dir, 'Images.npy')
        fCNNs = join(root_dir, 'CNN_Detections.npy')
        fAnnot = join(root_dir, 'Annotations.npy')
        return fImages, fCNNs, fAnnot

    def __getitem__(self, name):
        """ [ access ]
        """
        fImages, fCNNs, fAnnot = self.get_memmapped_file_names(name)
        self.create_memmapped_data(name)

        shape = self.get_video_shape(name)
        X = np.memmap(fImages, dtype='uint8', mode='r', shape=shape)

        n, m, h, w, _ = shape
        #CNN = np.memmap(fCNNs, dtype='uint8', mode='r', shape=
        #    (n, m, int(h/2), int(w/2), 14))

        Annot = self.get_annotations(name)
        assert len(Annot) == n,\
            'annotations and videos do not have the same number of cameras'
        for annot_per_cam in Annot:
            assert len(annot_per_cam) == m, \
                'annotations and videos do not have the same number of frames'

        #return X, CNN, Annot
        return X, Annot

    def get_video_shape(self, name):
        """ get the data shape for the given data
        """
        root_dir = join(self.root, 'marconi/' + name)
        image_dir = join(root_dir, join('Images', 'Images'))
        assert isdir(image_dir), 'Image directory must exist ->' + image_dir

        file_extension = MARCOnI.get_image_file_extension(name)  # jpg vs png
        Cam_Ids = sorted(list(
            set([f[6:8] for f in listdir(image_dir) if f.endswith(file_extension)])))

        Camera_Images = []
        m = -1  # number of frames per camera
        w, h = -1, -1  # width/height of videos
        for cam_id in Cam_Ids:
            Images = sorted([join(image_dir, f) for f in listdir(image_dir) if \
                f.endswith(file_extension) and f.startswith('frame_' + cam_id)])
            Camera_Images.append(Images)
            if m < 0:
                m = len(Images)
            else:
                assert m == len(Images)

            I = imread(Images[0])
            h_, w_, _ = I.shape
            if w < 0 and h < 0:
                w, h = w_, h_
            else:
                assert w == w_ and h == h_

        n = len(Cam_Ids)  # number of cameras
        return n, m, h, w, 3

    def create_memmapped_data(self, name):
        """ create the memory-mapped dataset for the given sub-set
        """
        root_dir = join(self.root, 'marconi/' + name)
        assert isdir(root_dir), 'directory ' + root_dir + " must exist!"

        file_extension = MARCOnI.get_image_file_extension(name)  # jpg vs png
        fImages, fCNNs, fAnnot = self.get_memmapped_file_names(name)

        # ----------------------------
        #     memory-mapped video
        # ----------------------------
        if not isfile(fImages):
            image_dir = join(root_dir, join('Images', 'Images'))
            assert isdir(image_dir), 'Image directory must exist ->' + image_dir

            Cam_Ids = sorted(list(
                set([f[6:8] for f in listdir(image_dir) if f.endswith('.' + file_extension)])))

            Camera_Images = []
            m = -1  # number of frames per camera
            w, h = -1, -1  # width/height of videos
            for cam_id in Cam_Ids:
                Images = sorted([join(image_dir, f) for f in listdir(image_dir) if \
                    f.endswith('.' + file_extension) and f.startswith('frame_' + cam_id)])
                Camera_Images.append(Images)
                if m < 0:
                    m = len(Images)
                else:
                    assert m == len(Images)

                I = imread(Images[0])
                h_, w_, _ = I.shape
                if w < 0 and h < 0:
                    w, h = w_, h_
                else:
                    assert w == w_ and h == h_

            n = len(Cam_Ids)  # number of cameras

            X = np.memmap(fImages, dtype='uint8', mode='w+', shape=(n, m, h, w, 3))

            for c, Images in enumerate(Camera_Images):
                I = np.array([imread(f) for f in Images], 'uint8')
                X[c] = I

            del X
        else:
            n, m, h, w, _ = self.get_video_shape(name)


        # ----------------------------
        #     CNN detections
        # ----------------------------
        # if not isfile(fCNNs):
        #     cnn_dir = join(root_dir, join('CNN_Detections', 'CNN_Detections'))
        #     assert isdir(cnn_dir), 'CNN-Detection folder must exist ->' + cnn_dir
        #
        #     Cam_Ids_cnn = sorted(list(
        #         set([f[0:8] for f in listdir(cnn_dir) if f.endswith('.png')])))
        #     assert len(Cam_Ids_cnn) == n
        #
        #     joints = ["j%03d.png" % (j,) for j in range(1, 15)]
        #
        #     CNNs = []
        #     for cam_id in Cam_Ids_cnn:
        #         Joints = []
        #         for j in joints:
        #             J = sorted([join(cnn_dir, f) for f in listdir(cnn_dir) if \
        #                 f.startswith(cam_id) and f.endswith(j)])
        #             assert len(J) == m
        #             Joints.append(J)
        #
        #         CNNs.append(Joints)
        #
        #     cnn_factor = MARCOnI.get_cnn_to_image_factor(name)
        #
        #     X = np.memmap(fCNNs, dtype='uint8', mode='w+', shape=
        #         (n, m, int(h/cnn_factor), int(w/cnn_factor), 14))
        #
        #     for c, cnn in enumerate(CNNs):
        #         for j, J in enumerate(cnn):
        #             J = np.array([imread(f) for f in J], 'uint8')
        #             X[c,:,:,:,j] = J
        #
        #     del X

    def download_set(self, name):
        """ download the given set
        """
        self.root_export = join(self.root, 'marconi/' + name)
        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + '/Images.zip'
        self.verbose = False
        self.download_and_unzip(url, zipfile_name='Images.zip',
            dest_folder='marconi/' + name + "/Images", dest_force=True,
            root_folder='marconi/' + name)

        # url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + '/CNN_Detections.zip'
        # self.download_and_unzip(url, zipfile_name='CNN_Detections.zip',
        #     dest_folder='marconi/' + name + "/CNN_Detections", dest_force=True,
        #     root_folder='marconi/' + name)

        self.root_export = join(self.root, 'marconi')
        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + \
            '/Annotations.mat'
        self.download_file(url, file_name='Annotations.mat', dest_folder=name)

        url = 'http://resources.mpi-inf.mpg.de/marconi/Data/' + name + \
            '/Calibration.dat'
        self.download_file(url, file_name='Calibration.dat', dest_folder=name)
        self.verbose = self._verbose
        utils.talk("\t" + name + " finished", self._verbose)
