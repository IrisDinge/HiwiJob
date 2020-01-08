import os
from PIL import Image
import shapely.geometry as shgeo


object_categories = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
              'swimming-pool', 'helicopter']


def parse_annotation(filename):
    """
    parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects=[]
    fd = []
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

def parse_annotation_2(filename):

    """
        parse the dota ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    :param filename:
    :return:
    """
    objects = parse_annotation(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    print(objects)
    return objects


def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1], poly[1][0], poly[1][1], poly[2][0], poly[2][1], poly[3][0], poly[3][1]]
    return outpoly



def cropout_object(filename, imagepath, size):
    """
    when area below 200, assume it lost the meaning already but should consider the size of original image
    so it will be modified by proportion
    :param filename:
    :param imagepath:
    :param size: setting inputsize
    :return:
    """


    im = Image.open(imagepath)
    objects = parse_annotation_2(filename)
    for obj in objects:
        poly = obj['poly']
        area = obj['area']
        print(area)

        if area < 200:
            continue
        else:
            xmin = int(min(poly[0], poly[2], poly[4], poly[6]))     # left
            ymin = int(min(poly[1], poly[3], poly[5], poly[7]))     # up
            xmax = int(max(poly[0], poly[2], poly[4], poly[6]))     # right
            ymax = int(max(poly[1], poly[3], poly[5], poly[7]))     # bottom
            subimage = im.crop((xmin, ymin, xmax, ymax))
            resizeimg = subimage.resize((size, size), Image.ANTIALIAS)
            resizeimg.show()

def crop_batches(root, label, img):
    """

    :param root:
    :param label:
    :param img:
    :return:
    """
    labelname = os.listdir(root + label)
    imgname = os.listdir(root + img)

    if len(labelname) != len(imgname):
        raise FileNotFoundError
    else:

        for i in range(len(labelname)):
            labeldir = os.path.join(root + label, labelname[i])
            imgdir = os.path.join(root + img, imgname[i])
            cropout_object(labeldir, imgdir, 200)



crop_batches('/home/dingjin/test/plane/', 'labelTxt', 'JPEGImages')












