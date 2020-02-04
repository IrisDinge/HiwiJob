import os, csv
from PIL import Image
import shapely.geometry as shgeo



class crop_dota(object):

    def __init__(self, config):
        self.filter_size = config['preparation']['filter_size']
        self.crop_out_size = config['preparation']['crop_out_size']
        self.root_dir = config['preparation']['root_dir']
        self.object_categories = config['preparation']['object_categories']
        self.image = 'images'
        self.label = 'labels'

    def parse_annotation(self, filename):
        """
        parse the dota ground truth in the format:
            [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        objects = []
        fd = []
        fd = open(filename, 'r')
        imgname = os.path.split(filename)[1]

        while True:
            line = fd.readline()
            safeline = line.strip().split(' ')
            object_dic = {}
            if line:
                if (len(safeline) < 9):
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

    def parse_annotation_2(self, filename):

        """
            parse the dota ground truth in the format:
            [x1, y1, x2, y2, x3, y3, x4, y4]
        :param filename:
        :return:
        """
        objects = self.parse_annotation(self, filename)
        for obj in objects:
            obj['poly'] = self.TuplePoly2Poly(self, obj['poly'])
            obj['poly'] = list(map(int, obj['poly']))
        # print(objects)
        return objects

    def TuplePoly2Poly(self, poly):
        outpoly = [poly[0][0], poly[0][1], poly[1][0], poly[1][1], poly[2][0], poly[2][1], poly[3][0], poly[3][1]]
        return outpoly

    def cropout_object(self, root_dir, filename, imagepath, size):
        """
        when area below 200, assume it lost the meaning already but should consider the size of original image
        so it will be modified by proportion
        :param filename:
        :param imagepath:
        :param size: setting inputsize
        :return:
        """

        im = Image.open(imagepath)
        objects = self.parse_annotation_2(self, filename)

        for obj in objects:
            poly = obj['poly']
            area = obj['area']
            id = obj['name']
            imgname = obj['imgname']

            if area < self.filter_size:  # whether area is too small to be useless
                continue
            else:
                xmin = int(min(poly[0], poly[2], poly[4], poly[6]))  # left
                ymin = int(min(poly[1], poly[3], poly[5], poly[7]))  # up
                xmax = int(max(poly[0], poly[2], poly[4], poly[6]))  # right
                ymax = int(max(poly[1], poly[3], poly[5], poly[7]))  # bottom
                subimage = im.crop((xmin, ymin, xmax, ymax))
                resizeimg = subimage.resize((size, size), Image.ANTIALIAS)
                resizeimg.save(root_dir + 'classification/' + id + '/' + imgname + '_' + str(int(area)) + '.png')

    def write_txt(self):
        f = open(self.root_dir + 'list.txt', 'w+')
        w = csv.writer(f, delimiter=' ')
        for path, dirs, files in os.walk(self.root_dir + 'classification/'):
            a = os.path.split(path)[1]
            for filename in files:
                w.writerow([path + '/' + filename] + [a])

    def crop_batches(self):
        """

        :param root:
        :param label:
        :param img:
        :return:
        """
        labelname = os.listdir(self.root_dir + self.label)
        imgname = os.listdir(self.root_dir + self.image)
        filename = []

        for i in labelname:
            name = os.path.splitext(i)[0]
            filename.append(name)

        for i in range(len(self.object_categories)):
            if not os.path.exists(self.root_dir + 'classification/' + self.object_categories[i]):
                os.makedirs(self.root_dir + 'classification' + '/' + self.object_categories[i])
            else:
                continue

        if len(labelname) != len(imgname):
            raise FileNotFoundError
        else:
            for i in range(len(filename)):
                labeldir = os.path.join(self.root_dir + self.label, filename[i] + '.txt')
                imgdir = os.path.join(self.root_dir + self.image, filename[i] + '.png')

                self.cropout_object(self.root_dir, labeldir, imgdir, self.crop_out_size)










