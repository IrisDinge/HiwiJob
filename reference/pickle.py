"""
from six.moves import cPickle as pickle
import numpy as np
import imageio


***Main Function***
1. get train/test.txt files
2. parse train/test files
3. generate train/test.pkl files 

***Additional Function***
the originla image data is going to be transformed in range of 0 to 1

***Usage***
pls change 'train' to 'test', and v.v



ROOT = '/home/dingjin/HiwiJob/example/'
class_id = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
            'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
            'swimming-pool', 'helicopter']


def normalize_image(x):
    
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val)/(max_val - min_val)
    return x


def train_image2numpy():
    img = []
    cls = []
    with open(ROOT + 'train.txt', 'r') as f:
        for line in f:
            image_path = line.split(" ", 1)[0]
            id = line.split(" ", 1)[1].strip()
            cls.append(id)
            img.append(imageio.imread(image_path))
            normalize_image(img)
    return img, cls

def test_image2numpy():
    img = []
    cls = []
    with open(ROOT + 'test.txt', 'r') as f:
        for line in f:
            image_path = line.split(" ", 1)[0]
            id = line.split(" ", 1)[1].strip()
            cls.append(id)
            img.append(imageio.imread(image_path))
            normalize_image(img)
    return img, cls

def normalize_class_id(class_id):
    """
    #Mapping class-id to int
    #:param x: a list of class-id
    #:Attention: DOTA contains 15 classes
    #:return: number of labels, number of class
    """
    dict_id = {}
    i = -1
    for item in class_id:
        if(i > 0 and item in dict_id):
            continue
        else:
            i = i + 1
            dict_id[item] = i

    label_list = []
    for item in class_id:
        label_list.append(dict_id[item])

    encoded = np.zeros((len(label_list), 15))

    for idx, val in enumerate(label_list):
        encoded[idx, val] = 1
    return encoded

def preprocess_and_save():
    features1 = train_image2numpy()[0]
    labels1 = normalize_class_id(train_image2numpy()[1])
    pickle.dump((features1, labels1), open(ROOT + 'train.pkl', 'wb'))
    features2 = test_image2numpy()[0]
    labels2 = test_image2numpy()[1]
    pickle.dump((features2, labels2), open(ROOT + 'test.pkl', 'wb'))

if __name__ == '__main__':
    preprocess_and_save()


"""
