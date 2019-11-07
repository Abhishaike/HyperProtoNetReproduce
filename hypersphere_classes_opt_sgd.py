#
# Obtain hyperspherical prototypes prior to network training.
#
# @inproceedings{mettes2016hyperspherical,
#  title={Hyperspherical Prototype Networks},
#  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
#  booktitle={Advances in Neural Information Processing Systems},
#  year={2019}
# }
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
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
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
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
        optimizer = optim.SGD([prototypes], lr=.01, momentum=.9)

    class_matched_points = dict(zip(unique_class_numbers, prototypes.detach().numpy()))
    return class_matched_points