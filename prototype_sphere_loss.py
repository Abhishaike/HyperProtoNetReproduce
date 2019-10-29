import torch
import scipy

def classification_loss(hypersphere_prediction, hypersphere_labels):
    cosine_similarity = (1 - torch.nn.functional.cosine_similarity(hypersphere_prediction, hypersphere_labels, dim=1, eps=1e-8)) ** 2
    mean_cosine_similarity = cosine_similarity.mean()
    return mean_cosine_similarity

def regression_loss(hypersphere_prediction, hypersphere_labels):
    pass

def joint_loss(optimized_points):
    pass