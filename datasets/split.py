import os
import codecs
import numpy as np
import math
from .dota_utils import getFilesfromCluster, custombasename, parse_annotation
import cv2

class splitbase():
    def __init__(self, dotapath, outpath, code = 'utf-8', gap=100,
                 subsize=1024, thresh=0.7, choosebestpoint=True,
                 ext = '.png'):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        """
        self.dotapath = dotapath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.dotapath, 'images')
        self.labelpath = os.path.join(self.dotapath, 'labels')
        self.outimagepath = os.path.join(self.outpath, 'imagesSub')
        self.outlabelpath = os.path.join(self.outpath, 'labelsSub')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outlabelpath):
            os.makedirs(self.outlabelpath)

    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = getFilesfromCluster(self.imagepath)
        imagenames = [custombasename(x) for x in imagelist if (custombasename(x) != 'Thumbs')]  # Linux hidden file just like .DS_Store
        for name in imagenames:
            self.SplitSingle(name, rate, self.ext)

    def SplitSingle(self, name, rate, extent):
        """
        :param name:
        :param rate: the resize scale for the image
        :param extent:
        :return:
        """
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = parse_annotation(fullname)
        for obj in objects:
            obj['poly'] = list(map(lambda x: rate*x, obj['poly']))
        if (rate != 1):
            # Need to be resized, scale up or down by rate
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:                                                                                                           # rate =1
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]
        left, up = 0, 0
        while (left<weight):
            if (left + self.subsize >= weight):                                                                         #subsize = 1024
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                




