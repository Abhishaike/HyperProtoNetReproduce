import torch
import scipy

def get_classification_loss(hypersphere_prediction, prototype_classification):
    cosine_similarity = (1 - torch.nn.functional.cosine_similarity(hypersphere_prediction, prototype_classification, dim=1, eps=1e-8)).pow(2)
    sum_cosine_similarity = cosine_similarity.sum()
    return sum_cosine_similarity

def get_regression_loss(prototype_regression, hypersphere_prediction, upper_bound_prototype):
    cosine_similarity = (prototype_regression - torch.nn.functional.cosine_similarity(hypersphere_prediction, upper_bound_prototype, dim=1, eps=1e-8)).pow(2)
    sum_cosine_similarity = cosine_similarity.sum()
    return sum_cosine_similarity