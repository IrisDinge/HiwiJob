import PIL
import os
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



if __name__=='__main__':
    getFilesfromCluster(dotapath='/Users/iris/Desktop/DOTA')