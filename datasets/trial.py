import PIL
import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import math



def parse_annotation(filedirs):
    """
    parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """

    #for files in os.walk(filename):

    #    line = f.readline()
    #    safeline = line.strip().split(' ')
    #    print(safeline)



parse_annotation(filedirs='/Users/iris/Desktop/DOTA/train/labels')
