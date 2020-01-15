import numpy as np
from six.moves import cPickle as pickle
import os
import sys
['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
              'swimming-pool', 'helicopter']

DEBUG = True
ROOT = '/data/vehicle-classification/dota/tmp'

TRAINSET = [
    ('train', 'plane'),
    ('train', 'baseball-diamond'),
    ('train', 'bridge'),
    ('train', 'ground-track-field'),
    ('train', 'small-vehicle'),
    ('train', 'large-vehicle'),
    ('train', 'ship'),
    ('train', 'tennis-court'),
    ('train', 'basketball-court'),
    ('train', 'storage-tank'),
    ('train', 'soccer-ball-field'),
    ('train', 'roundabout'),
    ('train', 'harbor'),
    ('train', 'swimming-pool'),
    ('train', 'helicopter'),
    ]

TESTSET = [
    ('test'),
    ]

if __name__=='__main__':
    print('Getting training files')
    train= []
    for (img_set, class_id) in TRAINSET:
        with open(f'{ROOT}/{img_set}/list.csv', 'r') as f:
            ids = f.read().strip().split()
        train += [f'{ROOT}/']
