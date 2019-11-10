from nn_losses import get_classification_loss, get_regression_loss
import torch
import torchvision.transforms as transforms
from nn_losses import get_classification_loss
import numpy as np
import scipy
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def train(model, device, train_loader, optimizer, epoch, classification_matched_points,
          operation, upper_bound_prototype, upper_bound, lower_bound, task_weight = .5):
    '''
    :param classification_matched_points: Preset prototypes associated with the style class
    :param operation: 'joint', 'classification', or 'regression'
    :param upper_bound_prototype: Preset prototypes associated with the regression year task
    :param upper_bound: largest year
    :param lower_bound: smallest year
    :param class_weight:
    :return:
    '''
    #repeat the upper bound prototype according to batch_size
    model.train()
    for batch_idx, (data, local_labels_year, local_labels_style) in enumerate(train_loader):
        upper_bound_prototype_tensor = torch.from_numpy(upper_bound_prototype).repeat(data.shape[0], 1).to(device)

        # get the prototypes associated with the class of the desired styles
        prototype_classification = torch.FloatTensor([list(classification_matched_points[class_num.item()]) for class_num in local_labels_style])
        image, prototype_classification, prototype_regression = data.to(device), prototype_classification.to(device), local_labels_year.to(device)
        optimizer.zero_grad()
        hypersphere_prediction = model(image)

        #choose loss function used based on operation
        if operation == 'regression':
            total_loss = get_regression_loss(prototype_regression, hypersphere_prediction, upper_bound_prototype_tensor)
        elif operation == 'classification':
            total_loss = get_classification_loss(hypersphere_prediction, prototype_classification)
        elif operation == 'joint':
            regression_loss = get_regression_loss(prototype_regression, hypersphere_prediction, upper_bound_prototype_tensor)
            classification_loss = get_classification_loss(hypersphere_prediction, prototype_classification)
            total_loss = (1. - task_weight) * regression_loss + task_weight * classification_loss

        total_loss.backward()
        optimizer.step()

        if batch_idx%50 == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item()))

            #style classification accuracy
            predicted_labels = assign_predicted_class(hypersphere_prediction.detach(), classification_matched_points) #get closest matching prototypes
            correct = (np.array(predicted_labels) == np.array(local_labels_style.cpu())) * 1
            print('Style Train Accuracy: ', correct.mean())


            #year mean error
            predicted_years = assign_regressed_value(hypersphere_prediction, upper_bound_prototype_tensor, upper_bound, lower_bound)
            mean_error = torch.abs(predicted_years - prototype_regression).mean()
            print('Year Train Mean Error: ', mean_error.item())


def test(model, device, test_loader, epoch, classification_matched_points,
          operation, upper_bound_prototype, upper_bound, lower_bound, task_weight):
    model.eval()
    all_correct_style = []
    all_correct_year = []
    all_loss = []
    with torch.no_grad():
        for batch_idx, (data, local_labels_year, local_labels_style) in enumerate(test_loader):
            upper_bound_prototype_tensor = torch.from_numpy(upper_bound_prototype).repeat(data.shape[0], 1).to(device)

            prototype_classification = torch.FloatTensor([list(classification_matched_points[class_num.item()]) for class_num in local_labels_style])
            image, prototype_classification, prototype_regression = data.to(device), prototype_classification.to(device), local_labels_year.to(device)
            hypersphere_prediction = model(image)

            # choose loss function used based on operation
            if operation == 'regression':
                total_loss = get_regression_loss(prototype_regression, hypersphere_prediction,
                                                 upper_bound_prototype_tensor)
            elif operation == 'classification':
                total_loss = get_classification_loss(hypersphere_prediction, prototype_classification)
            elif operation == 'joint':
                regression_loss = get_regression_loss(prototype_regression, hypersphere_prediction, upper_bound_prototype_tensor)
                classification_loss = get_classification_loss(hypersphere_prediction, prototype_classification)
                total_loss = (1. - task_weight) * regression_loss + task_weight * classification_loss

            all_loss.append(total_loss.item())

            #style mean error
            predicted_labels = assign_predicted_class(hypersphere_prediction.detach(), classification_matched_points)  # get closest matching prototypes
            correct = (np.array(predicted_labels) == np.array(local_labels_style.cpu())) * 1
            all_correct_style.append(correct.mean())

            # year mean error
            predicted_years = assign_regressed_value(hypersphere_prediction, upper_bound_prototype_tensor, upper_bound, lower_bound)
            mean_error = torch.abs(predicted_years - prototype_regression).mean()
            all_correct_year.append(mean_error.item())

    print('\nEpoch {0}, Test style accuracy: {1}, Test year error: {2}'.format(epoch,
                                                                              np.array(all_correct_style).mean(),
                                                                              np.array(all_correct_year).mean()))


def assign_predicted_class(hypersphere_prediction, class_matched_points):
    hypersphere_prediction = np.array(hypersphere_prediction.cpu())
    hypersphere_label, prototype_classification = list(class_matched_points.keys()), np.array(list(class_matched_points.values()))
    predicted_label = []
    for prediction in hypersphere_prediction: #for every prediction, find the closest prototype and get the label
        all_distances_scipy = [1-scipy.spatial.distance.cosine(prediction, label) for label in prototype_classification] #1-distance gives similarity
        predicted_label.append(hypersphere_label[np.argmax(all_distances_scipy)])
    return predicted_label


def assign_regressed_value(hypersphere_prediction, upper_bound_prototype, upper_bound, lower_bound):
    all_similarities = (torch.nn.functional.cosine_similarity(hypersphere_prediction, upper_bound_prototype)+1)/2.
    predicted_years = (all_similarities * (upper_bound - lower_bound)) + lower_bound
    return predicted_years

