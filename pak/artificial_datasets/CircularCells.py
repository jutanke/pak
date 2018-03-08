import numpy as np


class CircularCells:
    """
    Generate circular cells
    """

    def __init__(self, w, h):
        """

        :param w:
        :param h:
        """
        self.w = w; self.h = h

    def generate(self, cover):
        """

        :param cover: between 0..1 how much the
                    cells should cover
        :return:
        """
        w = self.w; h = self.h
        I = np.zeros((h,w))


        return I