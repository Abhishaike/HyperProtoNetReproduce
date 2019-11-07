import torchvision
import torchvision.transforms as transforms
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', 'data_loaders/data')))

def get_cifar_data(batch_size):
    trainloader = get_train_loader_cifar(batch_size)
    testloader = get_test_loader_cifar(batch_size)

    return trainloader, testloader

def get_train_loader_cifar(batch_size):
    print(os.getcwd())
    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, 4),
                                          transforms.ToTensor(),
                                          normalize])
    trainset = torchvision.datasets.CIFAR100(root='data/cifar-100-python/',
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True)
    return trainloader

def get_test_loader_cifar(batch_size):
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR100(root='data/cifar-100-python/',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False)
    return testloader
