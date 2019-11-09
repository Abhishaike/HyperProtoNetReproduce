import torchvision
import torchvision.transforms as transforms
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import sys

###code taken from https://github.com/psmmettes/hpn/blob/master/helper.py


def get_imagenet200_data(batch_size):
    basedir = "data/imagenet200/"

    mrgb = [0.485, 0.456, 0.406]
    srgb = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    train_data = datasets.ImageFolder(basedir + "train/",
                                      transform=transforms.Compose([transforms.RandomCrop(64, 4),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    normalize]))
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=True)

    test_data = datasets.ImageFolder(basedir + "test/",
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   normalize]))
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=batch_size,
                                             shuffle=False)

    return trainloader, testloader
