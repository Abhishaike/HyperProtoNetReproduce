import torch
import scipy

def get_classification_loss(hypersphere_prediction, prototype_classification):
    cosine_similarity = (1 - torch.nn.functional.cosine_similarity(hypersphere_prediction, prototype_classification, dim=1, eps=1e-8)).pow(2)
    sum_cosine_similarity = cosine_similarity.sum()
    return sum_cosine_similarity

def get_regression_loss(local_labels_year, hypersphere_prediction, upper_bound):
    cosine_similarity = (local_labels_year - torch.nn.functional.cosine_similarity(hypersphere_prediction, upper_bound, dim=1, eps=1e-8)).pow(2)
    sum_cosine_similarity = cosine_similarity.sum()
    return sum_cosine_similarity


def get_joint_loss(hypersphere_prediction, prototype_classification, class_weight = .5):
    classification_loss = get_classification_loss(hypersphere_prediction, prototype_classification)
    regression_loss = get_regression_loss(hypersphere_prediction, prototype_classification)
    total_loss = (1.-class_weight) * regression_loss + class_weight * classification_loss
    return total_loss
