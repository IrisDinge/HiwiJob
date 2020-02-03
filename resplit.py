import pandas as pd
from sklearn.model_selection import train_test_split

"""
***Main Function***
1. merge training set and validation set, then output all.csv: <image_path class-id>
2. split data set based on .csv, train and test proportion defined by parameter:test_size
3. output: 2 .csv files: train.txt and test.txt
    <image_path class-id >
"""

def merge_data(root):
    training_path = root + 'train/' + 'list.csv'
    validation_path = root + 'validation/' + 'list.csv'
    training_data = pd.read_csv(training_path, header=None, sep=' ')
    validation_data = pd.read_csv(validation_path, header=None, sep=' ')
    all = training_data.append(validation_data, ignore_index=True)
    all.to_csv(root + 'all.txt', sep=' ', header=None, index=False)

def split_dataset(root):
    merge_data(root)
    data = pd.read_csv(root + 'all.txt', header=None, sep=' ')
    x = data.iloc[:]
    x_train, x_test = train_test_split(x, test_size=0.3, random_state=0, )
    x_train.to_csv(root + 'train.txt', sep=' ', header=None, index=False)
    x_test.to_csv(root + 'test.txt', sep=' ', header=None, index=False)

if __name__ == '__main__':

    split_dataset('/home/dingjin/HiwiJob/example/')
