#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import types, os
import numpy as np
import PIL.Image
import yaml
import utils.dotasplit

class dota2():

    def __init__(self, config):
        print("Processing DOTA from", config.dota2.rootpath, "...")
        self.scale = config.dota2.scale
        self.cls = config.dota2.cls
        self.rootpath = config.dota2.rootpath
        self.path = config.dota2.outpath
        self.trainingpath = config.dota2.rootpath + "training/labelTxt/*.txt"
        self.validationpath = config.dota2.rootpath + "validation/labelTxt/*.txt"
        self.model = config.objectdetection_params.model
        self.gap = config.dota2.gap
        self.inputsize = config.dota2.inputsize
        self.num_process = config.dota2.num_process
        self.dirpath =config.dota2.rootpath + "training/"
        self.thresh = config.objectdetection_params.thresh
        self.cell = config.objectdetection_params.grid_cell
        self.anchor_box = config.objectdetection_params.anchors

    def parse_object(line):

        obj = types.SimpleNamespace()
        tokens = line.split(" ")
        obj.corners = [(float(tokens[2 * i]), float(tokens[2 * i + 1])) for i in range(4)]
        obj.cls = tokens[8]
        obj.difficult = False if tokens[9] == "0" else True
        return obj.__dict__

    def entrance(self):

        classID = "_".join(str(i)for i in self.cls)

        """
        if not os.path.isdir("data/dota2/" + classID):
            os.makedirs("data/dota2/" + classID)
        if not os.listdir(self.path + classID):
            split = utils.dotasplit.splitDOTA(self.dirpath, self.path, self.gap, self.inputsize, self.thresh, self.num_process)
            split.splitdata()
            dota2.run(self, classID)

            return self.cls
        else:
            files = os.listdir(self.path + classID)

            for file in files:
                if os.path.splitext(file)[-1] != ".npz":
                    raise TypeError("dataset DOTA with wrong format!")

                else:
                    for i in range(len(self.cls)):
                        id = file.split("_")[i]
                        if id not in self.cls:
                            raise NameError("selected class is not match!")

            print("dataset DOTA is ready to go!")
            return self.cls
        """
        #dota2.run(self, classID)
        #dota2.cover2YOLO(self)
        #dota2.pack(self)
        dota2.iou(self)
    def run(self, classID):

        files = list(glob.iglob(self.trainingpath))
        files.extend(glob.iglob(self.validationpath))
        print(files)

        d = {}
        for fn in files:
            key = fn.split("/")[-1].split(".")[0]

            with open(fn, "r") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line[-1] == "\n":
                    lines[i] = line[:-1]
            d[key] = {"path": fn, "source": lines[0].split(":")[1], "gsd": lines[1].split(":")[1], "objects": []}

            for line in lines[2:]:
                d[key]["objects"].append(dota2.parse_object(line))

        if not os.path.exists("cfgs/rawDOTA2.yml"):
            os.system(r"touch %s" % "cfgs/rawDOTA2.yml")

        with open("cfgs/rawDOTA2.yml", "w") as f:
            yaml.dump(d, f)

        del d
        del dota2.parse_object
        del files

        dota2.clean(self, classID)

    def clean(self, classID):

        with open("cfgs/rawDOTA2.yml", "r") as f:
            d = yaml.load(f, yaml.FullLoader)
        print("we have ", len(d), "instances now")
        remove = set()

        for key, values in d.items():
            if values["gsd"] == "null":
                remove.add(key)
            else:
                gsd = float(values["gsd"])
                if gsd < 0.01:
                    remove.add(key)

        for key in remove:
            path = d[key]["path"]
            gsd = d[key]["gsd"]
            print("Removing", gsd, "from", path)
            del d[key]

        del remove


        for k, v in d.items():
            path = v["path"][:-3] + "png"
            tokens = path.split("/")
            tokens[4] = "images"     ###default: 7 (may change depends on different directory)
            v["img_path"] = "/".join(tokens)
            img = PIL.Image.open(v["img_path"])
            v["bands"] = "".join(img.getbands())
            v["gsd"] = float(v["gsd"])
            v["size"] = img.size


        del img

        if not os.path.exists("cfgs/DOTA2.yml"):
            os.system(r"touch %s" % "cfgs/DOTA2.yml")

        with open("cfgs/DOTA2.yml", "w") as f:
            yaml.dump(d, f)

        dota2.cover2YOLO(self)

    def cover2YOLO(self):

        with open("cfgs/DOTA2.yml", "r") as f:
            d = yaml.load(f, yaml.FullLoader)

        d3 = {}
        for k, v in d.items():
            v = types.SimpleNamespace(**v)
            v.objects = [types.SimpleNamespace(**o) for o in v.objects if o["cls"] in self.cls]

            for i, o in enumerate(v.objects):
                x1, y1 = o.corners[0]
                x2, y2 = o.corners[1]
                x3, y3 = o.corners[2]
                x4, y4 = o.corners[3]
                img_w = v.size[0]
                img_h = v.size[1]
                xmax = max(x1, x2, x3, x4)
                xmin = min(x1, x2, x3, x4)
                ymax = max(y1, y2, y3, y4)
                ymin = min(y1, y2, y3, y4)
                x = (xmin + xmax)/2
                y = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
                o.boundingbox = x/img_w, y/img_h, w/img_w, h/img_h
                del o.corners
                v.objects[i] = vars(o)
            if len(v.objects) > 0:
                d3[k] = vars(v)
        print(len(d), len(d3))
        d = d3
        del d3

        if not os.path.exists("cfgs/DOTA2YOLO.yml"):
            os.system(r"touch %s" % "cfgs/DOTA2YOLO.yml")

        with open("cfgs/DOTA2YOLO.yml", "w") as f:
            yaml.dump(d, f)
        dota2.pack(self)

    def iou(self):

        print("!")


    def pack(self):
        classID = "_".join(str(i) for i in self.cls)
        with open("cfgs/DOTA2YOLO.yml", "r") as f:
            d = yaml.load(f, yaml.FullLoader)
            n = 0
        cls_map = {c: i for i, c in enumerate(self.cls)}
        #print(cls_map, len(self.cls))
        n = 0
        d2 = {}
        for k, v in d.items():
            v = types.SimpleNamespace(**v)
            v.objects = [types.SimpleNamespace(**o) for o in v.objects if o["cls"] in self.cls]
            if len(v.objects) > 0:
                d2[k] = v
        d = d2
        del d2
        for v in d.values():
            n += len(v.objects)
            print(n)


        train_x = np.empty((n, self.inputsize, self.inputsize, 3), dtype=np.float32)

        #class_probability = len(self.cls)
        confidence_score = 1
        #anchor_boxes = 9
        #n = anchor_boxes * (4 + 1 + class_probability)

        #train_y = np.empty((5, n, 128, 128), dtype=np.float32)
        #i = 0
        #for v in d.values():
        #    img = PIL.Image.open(v.img_path)
        #    for o in v.objects:
        #        train_y[i] = cls_map[o.cls] ################## IT IS NOT RIGHT
        #        bb = o.bounding_box
        #        img = np.asanyarray(img)
        #        if v.bands == "RGB":
        #            train_x[i] = img
        #        else:
        #            for c in range(3):
        #                train_x[i, :, :, c] = img
        #        i += 0

        #    np.savez_compressed("data/dota/" + classID + "/%s_scale_0.npz" % ("_".join(self.cls)),
        #                        x=train_x, y=train_y)
"""
        for k, v in d.items():
            v = types.SimpleNamespace(**v)
            v.objects = [types.SimpleNamespace(**o) for o in v.objects if o["cls"] in self.cls]
            for i, o in enumerate(v.objects):
                print(len(v.objects), i)
                boxes = np.array([], dtype=int)
                boxes = np.append(boxes, o.boundingbox[0])
                boxes = np.append(boxes, o.boundingbox[1])
                boxes = np.append(boxes, o.boundingbox[2])
                boxes = np.append(boxes, o.boundingbox[3])


            if len(v.objects) > 0:
                d2[k] = v
        print(len(d), len(d2))
        d = d2
        del d2

        #n = 0
        #for v in d.values():
        #    n += len(v.objects)

        
        #training_x = np.empty((1000, 3, self.inputsize, self.inputsize))
        # k Anchor boxes * (4 bounding box parameters + confidence score + class scores)
        # confidence_scores = pr(containing i object) * IoU(pred, truth)
        #
        #n = 9 * (4 + confidence_scores + class_scores)
        #training_y = np.empty((1000, n, 128, 128))
"""

