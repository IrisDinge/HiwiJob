import PIL
import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import math
"""
Assume /data/vehicle_classification contains three folders: train, validation, test
train and validation contain images, label while test only contain images
train and validation annotation should be horizontal bounding box just like in raw dataset
'imagesource': imagesource
'gsd': ground sample distance, the physical size of one image pixel, in meters
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
"""

def getFilesfromCluster(dotapath, ext=None):
    allfiles = []
    ExtensionFile = (ext != None)
    for root, dirs, files in os.walk(dotapath):
        """
        root: all directories
        dirs: train, validation
        files: images, labels
        allfiles: each file's entire directories
        """
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if ExtensionFile and extension in ext:
                allfiles.append(filepath)
            elif not ExtensionFile:
                allfiles.append(filepath)
    #print(allfiles)
    return allfiles

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def parse_annotation(filename):
    """
    parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects=[]
    fd = []
    print('filename:', filename)
    fd = open(filename, 'r')
    while True:
        line = fd.readline()
        safeline = line.strip().split(' ')
        object_dic = {}
        if line:
            if (len(safeline) < 9 ):
                continue
            if (len(safeline) >= 9):
                object_dic['name'] = safeline[8]
            if (len(safeline) == 9):
                object_dic['difficult'] = '0'
            elif (len(safeline) >= 10):
                object_dic['difficult'] = safeline[9]

            object_dic['poly'] = [(float(safeline[0]), float(safeline[1])),
                                  (float(safeline[2]), float(safeline[3])),
                                  (float(safeline[4]), float(safeline[5])),
                                  (float(safeline[6]), float(safeline[7]))]
            gtpoly = shgeo.Polygon(object_dic['poly'])
            object_dic['area'] = gtpoly.area
            objects.append(object_dic)
        else:
            break
    return objects




if __name__=='__main__':
    getFilesfromCluster(dotapath='/Users/iris/Desktop/DOTA')