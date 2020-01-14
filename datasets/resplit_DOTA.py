import pandas as pd
from sklearn.model_selection import train_test_split

"""
***Main Function***
1. merge training set and validation set, then output all.csv: <image_path class-id>
2. split data set based on .csv, train and test proportion defined by parameter:test_size



"""
def merge_data(root, training_path, validation_path):
    training_data = pd.read_csv(training_path, header=None, sep=' ')
    validation_data = pd.read_csv(validation_path, header=None, sep=' ')
    all = training_data.append(validation_data)
    all.to_csv('/home/dingjin/HiwiJob/example/all.csv', sep=' ', header=None, index=False)



def split_dataset(all_path):
    data = pd.read_csv(all_path)
    x, y = data.iloc[:, 1:], data.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)





if __name__ == '__main__':
    merge_data('/home/dingjin/HiwiJob/example/',
          '/home/dingjin/HiwiJob/example/train/list.csv',
          '/home/dingjin/HiwiJob/example/validation/list.csv')
    split_dataset('/home/dingjin/HiwiJob/example/all.csv')
