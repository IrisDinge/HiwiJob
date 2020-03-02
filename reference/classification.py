import datasets
import models
import torch.nn as nn

import torch.optim as optim
import os, sys
import torch
from torch.autograd import Variable
from utils.preparation import WarmUpLR
from tensorboardX import SummaryWriter


# TODO: loop for a certain number of epochs
# TODO: in each epoch loop, loop over the number of batches (for that, config needs to contain a mini-batch size and the dataset object needs a function which returns the dataset size)
# TODO: in each mini-batch iteration i with mini-batch size n: get training examples i*n to ((i+1)*n)-1 from dataset (i.e., that object needs yet another function for that purpose)
# TODO: pass each mini-batch to model object and get its prediction (i.e., model needs a function that takes a mini-batch and returns whatever the model currently predicts)
# TODO: calculate loss and update model parameters

class classification():

    def __init__(self, config):
        self.config = config.classificaion_params
        self.config_full = config
        os.environ['CUDA_VISIBLE_DEVICES'] = config.classificaion_params.gpu
        self.epoch = config.classificaion_params.epoch

        print("loading dataset", config.classificaion_params.dataset, "...")
        dataset = getattr(datasets, config.classificaion_params.dataset)(config)
        self.training_dataloader = dataset.get_training_dataloader()
        self.test_dataloader= dataset.get_test_dataloader()

        print("instantiating model", config.classificaion_params.model, "...")
        if config.classificaion_params.model == "MobileNetv2":
            model = models.MobileNetV2()
        elif self.config.model == 'Xception':
            model = models.Xception()
        else:
            print("The inputted network has not supported yet")
            sys.exit()

        self.net = model.cuda()

    def run(self):

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=5e-4)

        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config.milestones,
                                                         gamma=0.2)  # learning rate decay

        iter_per_epoch = len(self.training_dataloader)
        print(iter_per_epoch)

        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * self.config.warm)
        checkpoint_path = os.path.join(self.config.checkpoint_path, self.config.model, self.config.time_now)


        # use tensorboard

        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)
        writer = SummaryWriter(log_dir=os.path.join(
            self.config.log_dir, self.config.model, self.config.time_now))
        input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
        writer.add_graph(self.net, Variable(input_tensor, requires_grad=True))

        # create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

        best_acc = 0.0
        for epoch in range(1, self.config.epoch):
            if epoch > self.config.warm:
                train_scheduler.step(epoch)

            classification.train(self, warmup_scheduler, optimizer, loss_function, writer)
            acc = classification.eval_training(self, loss_function, writer)

        # start to save best performance model after learning rate decay to 0.01
            if epoch > self.config.milestones[1] and best_acc < acc:
                torch.save(self.net.state_dict(), checkpoint_path.format(net=self.config.model, epoch=epoch, type='best'))
                best_acc = acc
                continue

            if not epoch % self.config.save_epoch:
                torch.save(self.net.state_dict(), checkpoint_path.format(net=self.config.model, epoch=epoch, type='regular'))

        writer.close()


    def train(self, warmup_scheduler, optimizer, loss_function, writer):
        self.net.train()
        # print(net.train())
        for batch_index, (images, labels) in enumerate(self.training_dataloader):

            if self.config.epoch <= self.config.warm:
                warmup_scheduler.step()

            images = Variable(images)
            labels = Variable(labels)

            labels = labels.cuda()
            images = images.cuda()

            optimizer.zero_grad()
            outputs = self.net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            n_iter = (self.config.epoch - 1) * len(self.training_dataloader) + batch_index + 1

            last_layer = list(self.net.children())[-1]
            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=self.config.epoch,
                trained_samples=batch_index * self.config.batch_size + len(images),
                total_samples=len(self.training_dataloader.dataset)
            ))

            # update training loss for each iteration
            writer.add_scalar('Train/loss', loss.item(), n_iter)

        for name, param in self.net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, self.config.epoch)

    def eval_training(self, loss_function, writer):
        self.net.eval()

        test_loss = 0.0  # cost function error
        correct = 0.0

        for (images, labels) in self.test_dataloader:
            images = Variable(images)
            labels = Variable(labels)

            images = images.cuda()
            labels = labels.cuda()

            outputs = self.net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            test_loss / len(self.test_dataloader.dataset),
            correct.float() / len(self.test_dataloader.dataset)
        ))
        print()

        # add informations to tensorboard
        writer.add_scalar('Test/Average loss', test_loss / len(self.test_dataloader.dataset), self.config.epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(self.test_dataloader.dataset), self.config.epoch)


