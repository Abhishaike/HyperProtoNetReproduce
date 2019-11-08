import os
import sys
import argparse
import pandas as pd
import numpy as np
from   PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import transforms, datasets
import torchvision.models as models

## OMNIART LOADING CODE TAKEN FROM https://github.com/psmmettes/hpn

#
# Image loading.
#
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

#
# Image loading.
#
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

#
# Image loading.
#
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class OmniArtDataset(torch.utils.data.Dataset):

    #
    # Initialize the dataset with appropriate folders and transforms.
    #
    def __init__(self, root, datafile, transforms, c1=None, c2=None):
        self.root = root
        self.transform = transforms

        # Images and regression data.
        self.data = pd.read_csv(datafile)
        self.images = self.data["omni_id"].iloc[:]
        self.years = np.array(self.data["creation_year"].iloc[:])
        self.len = len(self.years)

        # Artist and style data (centuries not used in evaluation).
        self.centuries = np.array(self.data["century"].iloc[:])
        self.styles = np.array(self.data["school"].iloc[:])
        if c1 is None:
            c1 = np.unique(self.centuries)
            self.c1 = c1
        if c2 is None:
            c2 = np.unique(self.styles)
            self.c2 = c2

        # Assign names to class ids.
        self.centurylabels = np.zeros(self.len, dtype=int) - 1
        self.stylelabels = np.zeros(self.len, dtype=int) - 1
        self.toremove = []
        for i in range(self.len):
            aidx = np.where(self.centuries[i] == c1)[0]
            if len(aidx) == 1:
                self.centurylabels[i] = aidx[0]
            sidx = np.where(self.styles[i] == c2)[0]
            if len(sidx) == 1:
                self.stylelabels[i] = sidx[0]

    #
    # Get an example with the labels.
    #
    def __getitem__(self, index):
        image = self.root + str(self.images[index]) + ".jpg"
        image = pil_loader(image)
        image = self.transform(image)
        year = self.years[index]
        century = self.centurylabels[index]
        school = self.stylelabels[index]
        return (image, year, school)

    #
    # Dataset size.
    #
    def __len__(self):
        return self.len


#
# Load the complete dataset with classification and regression labels.
#
def load_omniart(basedir, trainfile, testfile, batch_size):
    # Transformations.
    mrgb = [0.485, 0.456, 0.406]
    srgb = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mrgb, std=srgb)
    ])

    # Train set.
    trainset = OmniArtDataset(basedir + "train/", trainfile, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
                                              shuffle=True)

    # Test set.
    c1, c2 = trainset.c1, trainset.c2
    testset = OmniArtDataset(basedir + "test/", testfile, transform, c1, c2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, \
                                             shuffle=True)

    return trainloader, testloader

def load_regression_prototypes():
    pass