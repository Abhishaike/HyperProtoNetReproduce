import torchvision
import torchvision.transforms as transforms
import torch


def get_train_loader():
    transform_train = [transforms.RandomHorizontalFlip(),
                         transforms.RandomCrop(32, padding=4),
                         transforms.ToTensor()]
    trainset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=True,
                                             transform=[transform_train])
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size_train,
                                              shuffle=True,
                                              num_workers=4)

def get_test_loader():
    transform_test = [transforms.ToTensor()]
    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size_test,
                                             shuffle=False,
                                             num_workers=4)