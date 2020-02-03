import os
import argparse
from utils.data_envs import initEnv
from datasets.crop_out import crop_dota
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, help='dataset_name')
    parser.add_argument('-rs', type=bool, default=False, help='re-split dataset when it needs')
    parser.add_argument('-dl', type=bool, default=False, help='build custom data_loader')
    args = parser.parse_args()

    config = initEnv(args.dataset)
    print(config)

