#
# Obtain hyperspherical prototypes prior to network training.
#
# Portions of code taken from https://github.com/psmmettes/hpn/blob/master/helper.py
#
import os
import sys
import numpy as np
import random
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn


def combined_loss(prototypes, num_classes, output_dimension, unique_classes):
    cosine_similarity = prototype_loss(prototypes)
    privilege_info = privilege_info_loss(prototypes, num_classes, output_dimension, unique_classes)
    return cosine_similarity + privilege_info

def privilege_info_loss(prototypes, num_classes, output_dimension, unique_classes):
    '''
    Insert the loss function for the priv info here.
    '''
    return 0

# Compute the loss related to the prototypes.
#
def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    # Remove diag from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()

#
# Main entry point of the script.
#
def create_hypersphere_loss_w_sgd(num_classes, output_dimension, unique_class_numbers):
    prototypes = torch.randn(num_classes, output_dimension)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1), requires_grad = True)
    optimizer = optim.SGD([prototypes], lr=.01, momentum=.9)

    # Optimize for separation.
    for epoch in range(2000):
        # Compute loss.
        loss, sep = prototype_loss(prototypes)
        # Update.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {0}: {1}".format(epoch, sep))
        # Renormalize prototypes.
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1), requires_grad = True)
        optimizer = optim.SGD([prototypes], lr=.01, momentum=.9)

    class_matched_points = dict(zip(unique_class_numbers, prototypes.detach().numpy()))
    return class_matched_points