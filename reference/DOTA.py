#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import types, os
import numpy as np
import PIL.Image
import yaml

class DOTA():
    def __init__(self, config):
        print("Processing DOTA from", config.DOTA.path, "...")
        self.scale = config.DOTA.scale
        self.cls = config.DOTA.cls
        self.path = config.DOTA.path
        self.trainingpath = config.DOTA.path + "training/labelTxt/*.txt"
        self.validationpath = config.DOTA.path + "validation/labelTxt/*.txt"

    def parse_object(line):
        obj = types.SimpleNamespace()
        tokens = line.split(" ")
        obj.corners = [(float(tokens[2 * i]), float(tokens[2 * i + 1])) for i in range(4)]
        obj.cls = tokens[8]
        obj.difficult = False if tokens[9] == "0" else True
        return obj.__dict__

    def run(self):
        files = list(glob.iglob(self.trainingpath))
        files.extend(glob.iglob(self.validationpath))
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
                d[key]["objects"].append(DOTA.parse_object(line))


        if not os.path.exists("cfgs/DOTA.yml"):
            os.system(r"touch %s" % "cfgs/DOTA.yml")

        with open("cfgs/DOTA.yml", "w") as f:
            yaml.dump(d, f)

        del d
        del DOTA.parse_object
        del files

    def clean(self):

        with open("cfgs/DOTA.yml", "r") as f:
            d = yaml.load(f, yaml.FullLoader)
        print(len(d))

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
        print(len(d))

        for k, v in d.items():
            path = v["path"][:-3] + "png"
            tokens = path.split("/")
            tokens[5] = "images"     ###default: 7 (may change depends on different directory)
            v["img_path"] = "/".join(tokens)
            img = PIL.Image.open(v["img_path"])
            v["bands"] = "".join(img.getbands())
            v["gsd"] = float(v["gsd"])

        del img

        if not os.path.exists("cfgs/DOTA_cleaned.yml"):
            os.system(r"touch %s" % "cfgs/DOTA_cleaned.yml")

        with open("cfgs/DOTA_cleaned.yml", "w") as f:
            yaml.dump(d, f)

    def pack(self):

        with open("cfgs/DOTA_cleaned.yml", "r") as f:
            d = yaml.load(f, yaml.FullLoader)
        cls_map = {c: i for i, c in enumerate(self.cls)}
        d2 = {}
        for k, v in d.items():
            v = types.SimpleNamespace(**v)
            v.objects = [types.SimpleNamespace(**o) for o in v.objects if o["cls"] in self.cls]
            for i, o in enumerate(v.objects):
                o.corners = np.asanyarray(o.corners, dtype=np.int32)
                o.bounding_box = np.empty((2, 2), dtype=np.int32)
                o.bounding_box[0] = np.min(o.corners, axis=0)
                o.bounding_box[1] = np.max(o.corners, axis=0)
                v.objects[i] = o
            if len(v.objects) > 0:
                d2[k] = v
        print(len(d), len(d2))
        d = d2
        del d2

        n = 0
        for v in d.values():
            n += len(v.objects)

        for k in range(2, 9):
            patch_size = 16 * k

            train_x = np.empty((n, patch_size, patch_size, 3), dtype=np.uint8)
            train_y = np.empty(n, dtype=np.int8)

            i = 0
            for v in d.values():
                img = PIL.Image.open(v.img_path)
                for o in v.objects:
                    train_y[i] = cls_map[o.cls]
                    bb = o.bounding_box
                    img_o = img.crop((bb[0, 0], bb[0, 1], bb[1, 0], bb[1, 1]))
                    img_o = img_o.resize((patch_size, patch_size), resample=PIL.Image.BICUBIC)
                    img_o = np.asanyarray(img_o)
                    if v.bands == "RGB":
                        train_x[i] = img_o
                    else:
                        for c in range(3):
                            train_x[i, :, :, c] = img_o
                    i += 0
                np.savez_compressed("data/dota/dota_%s_%i_scale_0.npz" % ("_".join(self.cls), patch_size), x=train_x, y=train_y)

        del train_x
        del train_y
        del i
        del n

        factor = 0.5 * (2 * self.scale - 1)
        n = 0
        for v in d.values():
            n += len(v.objects)
        for k in range(2, 9):
            patch_size = 16 * k

            train_x = np.empty((n, patch_size, patch_size, 3), dtype=np.uint8)
            train_y = np.empty(n, dtype=np.int8)

            i = 0
            for v in d.values():
                img = PIL.Image.open(v.img_path)
                for o in v.objects:
                    train_y[i] = cls_map[o.cls]
                    bb = np.copy(o.bounding_box)
                    size = factor * (bb[1] - bb[0])
                    bb[0] = np.maximum(bb[0] - size, (0, 0))
                    bb[1] = np.minimum(bb[1] + size, img.size)
                    img_o = img.crop(bb[0, 0], bb[0, 1], bb[1, 0], bb[1, 1])
                    img_o =img_o.resize((patch_size, patch_size), resample = PIL.Image.BICUBIC)
                    img_o = np.array(img_o)
                    if v.bands == "RGB":
                        train_x[i] = img_o
                    else:
                        for c in range(3):
                            train_x[i, :, :, c] = img_o
                    i += 1
            np.savez_compressed("data/DOTA/DOTA_%s_%i_scale_%i.npz" % ("_".join(self.cls), patch_size, self.scale), x=train_x, y=train_y)

            del size
            del bb
            del train_x
            del train_y
            del i
            del n
            del factor



































