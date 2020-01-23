import argparse
from utils.envs import initEnv
import torch.nn as nn
import torch.optim as optim
import os
import torch
from torch.autograd import Variable
import logging as log
from pprint import pformat
#from utils.hyperparam import HyperParams
#from utils.preparation import *
#from tasks import classification

from tensorboardX import SummaryWriter
from utils.preparation import get_training_dataloader, get_network, get_test_dataloader, WarmUpLR
from config import settings


def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def eval_training(epoch):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='model name', default=None)
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    args = parser.parse_args()
    config = initEnv(args.model_name, train_flag=1)

    net = get_network(args, use_gpu=args.gpu)


    cifar100_training_loader = get_training_dataloader(
        config['CIFAR100_TRAIN_MEAN'],
        config['CIFAR100_TRAIN_STD'],
        num_workers=config['train']['nworkers'],
        batch_size=config['train']['batch_size'],
        shuffle=config['shuffle']
    )

    cifar100_test_loader = get_test_dataloader(
        config['CIFAR100_TRAIN_MEAN'],
        config['CIFAR100_TRAIN_STD'],
        num_workers=config['test']['nworkers'],
        batch_size=config['test']['batch_size'],
        shuffle=config['shuffle']
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['MILESTONES'],
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(config['CHECKPOINT_PATH'], args.model_name, config['TIME_NOW'])

 # use tensorboard
    if not os.path.exists(config['LOG_DIR']):
        os.mkdir(config['LOG_DIR'])
    writer = SummaryWriter(log_dir=os.path.join(
        config['LOG_DIR'], args.model_name, config['TIME_NOW']))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0

    for epoch in range(1, config['EPOCH']):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > config['MILESTONES'][1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % config['SAVE_EPOCH']:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
