import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import *
from torch.utils.data import DataLoader, random_split


class LoadData:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, dataset, val_size: int = 100):
        self.train_size = 10000
        self.val_size = val_size
        self.dataset = dataset

        self.dataset_train, self.dataset_test = self.download_dataset()
        self.pool_size = self.dataset_train.data.shape[0] - self.train_size - self.val_size

        (
            self.dataset_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.dataset_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data

    def check_dataset_folder(self) -> bool:
        """Check whether MNIST folder exists, skip download if existed"""
        if os.path.exists("datasets/"+self.dataset):
            return False
        return True

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""
        if 'MNIST' in self.dataset.upper():
            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            transform_test = transform_train
        if 'CIFAR' in self.dataset.upper():
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        download = self.check_dataset_folder()
        dataset_list = ['LSUN', 'LSUNClass',
        'ImageFolder', 'DatasetFolder', 'FakeData',
        'CocoCaptions', 'CocoDetection',
        'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
        'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
        'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
        'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
        'Caltech101', 'Caltech256', 'CelebA', 'WIDERFace', 'SBDataset',
        'VisionDataset', 'USPS', 'Kinetics400', 'HMDB51', 'UCF101',
        'Places365']
        func_dict = {'STL10': STL10, 'MNIST': MNIST, 'CIFAR100': CIFAR100, 'CIFAR10': CIFAR10}
        dataset_train = func_dict.get(self.dataset)("datasets/", train=True, download=download, transform=transform_train)
        dataset_test = func_dict.get(self.dataset)("datasets/", train=False, download=download, transform=transform_test)
        return dataset_train, dataset_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.dataset_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=self.dataset_test, batch_size=10000, shuffle=True
        )
        dataset_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        dataset_test, y_test = next(iter(test_loader))
        return dataset_train_All, y_train_All, X_val, y_val, X_pool, y_pool, dataset_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            dataset_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(10):
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=2, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.dataset_train_All[initial_idx]
        y_init = self.y_train_All[initial_idx]
        print(f"Initial training data points: {X_init.shape[0]}")
        print(f"Data distribution for each class: {np.bincount(y_init)}")
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.dataset_test),
            self.tensor_to_np(self.y_test),
        )
