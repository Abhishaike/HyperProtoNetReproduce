import torchvision
import torchvision.transforms as transforms
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

def get_mnist_data(batch_size):
    trainloader = get_train_loader_mnist(batch_size)
    testloader = get_test_loader_mnist(batch_size)

    return trainloader, testloader

def get_train_loader_mnist(batch_size):
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                         #transforms.RandomCrop(32, padding=4),
                         transforms.ToTensor()])
    trainset = datasets.MNIST(root='./mnist',
                             train=True,
                             download=True,
                             transform=transform_train)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True)
    return trainloader

def get_test_loader_mnist(batch_size):
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root='./mnist',
                            train=False,
                            download=True,
                            transform=transform_test)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False)
    return testloader


def get_cifar_data(batch_size):
    trainloader = get_train_loader_cifar(batch_size)
    testloader = get_test_loader_cifar(batch_size)

    return trainloader, testloader

def get_train_loader_cifar(batch_size):
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                         #transforms.RandomCrop(32, padding=4),
                         transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True)
    return trainloader

def get_test_loader_cifar(batch_size):
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False)
    return testloader
