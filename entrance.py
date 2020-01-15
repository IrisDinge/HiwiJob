from six.moves import cPickle as pickle
import csv
import sys
sys.path.insert(0, '.')

"""
***Main Function***
1. get train/test files
2. parse train/test files
3. generate train/test files 
"""

ROOT = '/home/dingjin/HiwiJob/example/'
txt_file = ['train', 'test']


#def search_image():
#    root_dir = ROOT
#    return f'{root_dir}/{csv_file}/'




if __name__ == '__main__':
    print("Getting Dataset")
    train = []
    with open(f'{ROOT}/train.txt', 'r') as f:
        image_path = f.readlines()
        print(image_path)









"""
def pickle_from_csv(root):

    for i in range(len(csv_file)):
        csv_path = root + csv_file[i] + '.csv'
        pkl_path = root + csv_file[i] + '.csv'
        print(csv_path, pkl_path)
        x = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:x.append(line)
        with open(pkl_path, 'wb') as f:
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)



pickle_from_csv('/home/dingjin/HiwiJob/example/')

"""
