import copy
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

from lib.loading_data.poison_data import PoisonTestDataset


class LoadData:
    def __init__(self, data_name, poison_type):
        self.data_name = data_name
        self.poison_type = poison_type

    def get_date(self, batch_size=64):
        if self.data_name == 'cifar10':
            train_set, test_loader, poison_test_loader = self.get_cifar10(batch_size)
            return train_set, test_loader, poison_test_loader
        elif self.data_name == 'cifar100':
            train_set, test_loader, poison_test_loader = self.get_cifar100(batch_size)
            return train_set, test_loader, poison_test_loader
        elif self.data_name == 'mnist':
            train_set, test_loader, poison_test_loader = self.get_mnist(batch_size)
            return train_set, test_loader, poison_test_loader
        elif self.data_name == 'tiny_imagenet':
            train_set, test_loader, poison_test_loader = self.get_tiny_imagenet(batch_size)
            return train_set, test_loader, poison_test_loader
        else:
            raise KeyError(f'{self.data_name} not supported')

    def get_cifar10(self, batch_size=64):
        # 加载数据集
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

        poison_test_set = PoisonTestDataset(copy.deepcopy(test_set),
                                            dataset_name='cifar10',
                                            poison_type=self.poison_type)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        poison_test_loader = torch.utils.data.DataLoader(poison_test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_set, test_loader, poison_test_loader

    def get_cifar100(self, batch_size=64):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # transform_standard = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        # ])

        train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform_train)
        # train_original_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
        #                                                   transform=transform_standard)
        # train_combined_set = ConcatDataset([train_original_set, train_set])
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        poison_test_set = PoisonTestDataset(copy.deepcopy(test_set),
                                            dataset_name='cifar100',
                                            poison_type=self.poison_type)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        poison_test_loader = torch.utils.data.DataLoader(poison_test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_set, test_loader, poison_test_loader

    def get_mnist(self, batch_size=64):
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST('./data',
                                                   train=True,
                                                   download=True,
                                                   transform=trans_mnist)
        test_dataset = torchvision.datasets.MNIST('./data',
                                                  train=False,
                                                  download=True,
                                                  transform=trans_mnist)

        poison_test_set = PoisonTestDataset(copy.deepcopy(test_dataset),
                                            dataset_name='mnist',
                                            poison_type=self.poison_type)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        poison_test_loader = torch.utils.data.DataLoader(poison_test_set, batch_size=batch_size, shuffle=False, num_workers=8)

        return train_dataset, test_loader, poison_test_loader

    def get_tiny_imagenet(self, batch_size=64):
        # 定义数据预处理
        transform_train = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        # transform_standard = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        # 加载数据集
        train_set = ImageFolder(root='data/tiny-imagenet/train', transform=transform_train)
        # train_original_set = ImageFolder(root='data/tiny-imagenet/train', transform=transform_standard)
        # train_combined_set = ConcatDataset([train_original_set, train_set])

        test_dataset = ImageFolder(root='data/tiny-imagenet/val', transform=transform_test)
        poison_test_set = PoisonTestDataset(copy.deepcopy(test_dataset), dataset_name='tiny_imagenet', poison_type=self.poison_type)

        num_workers = os.cpu_count() // 2
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        poison_test_loader = torch.utils.data.DataLoader(poison_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_set, test_loader, poison_test_loader
