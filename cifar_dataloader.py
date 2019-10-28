import torchvision
import torchvision.transforms as transforms
import torch.utils

def get_cifar_data(batch_size):
    trainloader = get_train_loader(batch_size)
    testloader = get_test_loader(batch_size)

    return trainloader, testloader

def get_train_loader(batch_size):
    transform_train = [transforms.RandomHorizontalFlip(),
                         transforms.RandomCrop(32, padding=4),
                         transforms.ToTensor()]
    trainset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=True,
                                             transform=[transform_train])
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1)
    return trainloader

def get_test_loader(batch_size):
    transform_test = [transforms.ToTensor()]
    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=1)
    return testloader
