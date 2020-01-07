import os
import codecs
import numpy as np
import math
from .dota_utils import getFilesfromCluster, custombasename, parse_annotation
import cv2
import shapely.geometry as shgeo
import copy


classid_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
              'swimming-pool', 'helicopter']

def choose_best_point(poly1, poly2):
    """
    
    :param poly1: positive
    :param poly2: negative
    :return: smallest distance is the best one
    """
    
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distance = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate]) # Looking for the best one in poly1 that fit poly2
    sorted = distance.argsort()  # sort order of index from small to big
    
    return combinate[sorted[0]] 


def cal_line_length(point1, point2):
    """
    
    :param point1: 
    :param point2: 
    :return: 
    """
    length = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)) # distance = sqrt[(x1 - x2)^2 + (y1 -y2)^2]
    return length
    
    
    
    
    
    
    
class splitbase():
    def __init__(self, dotapath, outpath, code = 'utf-8', gap=100,
                 subsize=1024, thresh=0.7, choosebestpoint=True,                                                        #Change subsize
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

    def polyorig2sub(self, left, up, poly):
        """
        decide on where should the sub. QUITE CONFUSED
        :param left:
        :param up:
        :param poly:
        :return:
        """
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i*2] = int(poly[i*2]-left)
            polyInsub[i*2 + 1] = int(poly[i*2 + 1] -up)
        return polyInsub


    def calhalf_iou(self, poly1, poly2):
        """
            It is not usual iou, this one is value of intersection over polygones 1

        """
        intersection_poly = poly1.intersection(poly2)                                                                          #intersection between poly1 and poly2
        intersection_area = intersection_poly.area
        poly1_area = poly1.area
        half_iou = intersection_area / poly1_area

        return intersection_poly, half_iou

    def saveimagpathces(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left:(left + self.subsize)])         #Here determine the consist of subimagename
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imwrite(outdir, subimg)         # (directory, which image)
        

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outimagepath, subimgname + '.txt')
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down), (left, down)])

        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'[0]], obj['poly'][1]),
                                        (obj['poly'[2]], obj['poly'][3]),
                                        (obj['poly'[4]], obj['poly'][5]),
                                        (obj['poly'[6]], obj['poly'][7])])

                if (gtpoly.area <= 0):
                    continue
                inter_poly, half_iou = self.calhalf_iou(gtpoly, imgpoly)

                print('Writing...')
                if(half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                elif(half_iou > 0):
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1) # return a properly oriented copy of the given polygon
                    # sign=1 means that the coordinates of the product's exterior ring will be oriented counter-clockwise
                    
                    





    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = getFilesfromCluster(self.imagepath)
        imagenames = [custombasename(x) for x in imagelist if (custombasename(x) != 'Thumbs')]                          #Linux hidden file just like .DS_Store
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
        else:                                                                                                           #rate =1
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'                                                                    #modify name Part1
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
                subimgname = outbasename + str(left) + '___' + str(up)                                                  #modify name Part2
                self.savepathches(resizeimg, objects, subimgname, left, up, right, down)


                




