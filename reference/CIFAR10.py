import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CIFAR10():


    def __init__(self, config):
        self.mean = config.CIFAR10.mean
        self.std = config.CIFAR10.std
        self.batch_size = config.CIFAR10.batch_size
        self.num_workers = config.CIFAR10.num_workers
        print("CIFAR10 loading from", config.CIFAR10.path)

    def get_training_dataloader(self, shuffle=True):
        """ return training dataloader
        Args:
            mean: mean of cifar100 training dataset
            std: std of cifar100 training dataset
            path: path to cifar100 training python dataset
            batch_size: dataloader batchsize
            num_workers: dataloader num_works
            shuffle: whether to shuffle
        Returns: train_data_loader:torch dataloader object
        """
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                          transform=transform_train)
        cifar100_training_loader = DataLoader(
            cifar100_training, shuffle=shuffle, num_workers=self.num_workers, batch_size=self.batch_size)

        return cifar100_training_loader

    def get_test_dataloader(self, shuffle=True):
        """ return training dataloader
        Args:
            mean: mean of cifar100 test dataset
            std: std of cifar100 test dataset
            path: path to cifar100 test python dataset
            batch_size: dataloader batchsize
            num_workers: dataloader num_works
            shuffle: whether to shuffle
        Returns: cifar100_test_loader:torch dataloader object
        """

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                      transform=transform_test)
        cifar100_test_loader = DataLoader(
            cifar100_test, shuffle=shuffle, num_workers=self.num_workers, batch_size=self.batch_size)

        return cifar100_test_loader

