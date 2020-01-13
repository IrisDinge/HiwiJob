import os, csv
from PIL import Image
import shapely.geometry as shgeo
import pandas as pd



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
    imgname = os.path.split(filename)[1]

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
            object_dic['imgname'] = os.path.splitext(imgname)[0]
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
    #print(objects)
    return objects


def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1], poly[1][0], poly[1][1], poly[2][0], poly[2][1], poly[3][0], poly[3][1]]
    return outpoly



def cropout_object(root, filename, imagepath, size):
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
        id = obj['name']
        imgname = obj['imgname']


        if area < 10:                                              # whether area is too small to be useless
            continue
        else:
            xmin = int(min(poly[0], poly[2], poly[4], poly[6]))     # left
            ymin = int(min(poly[1], poly[3], poly[5], poly[7]))     # up
            xmax = int(max(poly[0], poly[2], poly[4], poly[6]))     # right
            ymax = int(max(poly[1], poly[3], poly[5], poly[7]))     # bottom
            subimage = im.crop((xmin, ymin, xmax, ymax))
            resizeimg = subimage.resize((size, size), Image.ANTIALIAS)
            resizeimg.save(root + 'classification/' + id + '/' + imgname + '_' + str(int(area)) + '.png')
                #resizeimg.show()




def write_csv(root):
    f = open(root + 'list.csv', 'w+')
    w = csv.writer(f)
    for path, dirs, files in os.walk(root + 'classification/'):
        a = os.path.split(path)[1]
        for filename in files:
            w.writerow([path+'/'+filename]+[a])


def crop_batches(root, label, img):
    """

    :param root:
    :param label:
    :param img:
    :return:
    """
    labelname = os.listdir(root + label)
    imgname = os.listdir(root + img)
    filename = []

    for i in labelname:
        name = os.path.splitext(i)[0]
        filename.append(name)

    for i in range(len(object_categories)):
        if not os.path.exists(root + 'classification/' + object_categories[i]):
            os.makedirs(root + 'classification' + '/' + object_categories[i])
        else:
            continue


    if len(labelname) != len(imgname):
        raise FileNotFoundError
    else:
        for i in range(len(filename)):


            labeldir = os.path.join(root + label, filename[i] + '.txt')
            imgdir = os.path.join(root + img, filename[i] + '.png')

            cropout_object(root, labeldir, imgdir, 10)



crop_batches('/Users/iris/Desktop/DOTA/train/', 'labels', 'images')
write_csv('/Users/iris/Desktop/DOTA/train/')


