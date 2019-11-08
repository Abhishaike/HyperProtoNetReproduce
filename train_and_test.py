from nn_losses import regression_loss, classification_loss, joint_loss
import torch
import torchvision.transforms as transforms
from nn_losses import classification_loss
import numpy as np
import scipy
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def train(model, device, train_loader, optimizer, epoch, classification_matched_points):
    model.train()
    for batch_idx, (local_batch, local_labels) in enumerate(train_loader):
        # Transfer to GPU
        hypersphere_vector = torch.FloatTensor([list(classification_matched_points[class_num.item()]) for class_num in local_labels]) #get the vector of the label
        image, hypersphere_vector = local_batch.to(device), hypersphere_vector.to(device)
        optimizer.zero_grad()
        hypersphere_prediction = model(image)
        cosine_loss = classification_loss(hypersphere_prediction, hypersphere_vector)
        cosine_loss.backward()
        optimizer.step()
        if batch_idx%50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cosine_loss.item()))
            hypersphere_labels = assign_predicted_class(hypersphere_prediction.detach(), classification_matched_points) #get closest matching prototypes
            correct = (np.array(hypersphere_labels) == np.array(local_labels.cpu())) * 1
            print('Train Accuracy: ', correct.mean())

        #cv2.imshow('hi',image[5].cpu().numpy().transpose(1, 2, 0)) #see example batch image


def test(model, device, test_loader, class_matched_points, epoch):
    model.eval()
    all_correct = []
    all_loss = []
    with torch.no_grad():
        for batch_idx, (local_batch, local_labels) in enumerate(test_loader):
            hypersphere_vector = torch.FloatTensor([list(class_matched_points[class_num.item()]) for class_num in local_labels]).to(device)  # get the vector of the label
            image, hypersphere_labels = local_batch.to(device), local_labels.to(device)
            hypersphere_prediction = model(image)
            cosine_loss = classification_loss(hypersphere_prediction, hypersphere_vector)
            pred_labels = assign_predicted_class(hypersphere_prediction, class_matched_points) #get closest matching prototypes
            correct = (np.array(hypersphere_labels.cpu()) == pred_labels)
            all_correct.extend(correct * 1)
            all_loss.append(cosine_loss.item())
    print('\n Epoch {0}, Test set accuracy: {1}, Loss: {2}'.format(epoch, np.array(all_correct).mean(), np.array(all_loss).mean()))


def assign_predicted_class(hypersphere_prediction, class_matched_points):
    hypersphere_prediction = np.array(hypersphere_prediction.cpu())
    hypersphere_label, hypersphere_vector = list(class_matched_points.keys()), np.array(list(class_matched_points.values()))
    predicted_label = []
    for prediction in hypersphere_prediction: #for every prediction, find the closest prototype and get the label
        all_distances_scipy = [1-scipy.spatial.distance.cosine(prediction, label) for label in hypersphere_vector] #1-distance gives similarity
        predicted_label.append(hypersphere_label[np.argmax(all_distances_scipy)])
    return predicted_label


def assign_regressed_value(hypersphere_prediction, upper_bound, lower_bound):
    hypersphere_prediction = np.array(hypersphere_prediction.cpu())
    hypersphere_label, hypersphere_vector = list(class_matched_points.keys()), np.array(list(class_matched_points.values()))
    predicted_label = []
    for prediction in hypersphere_prediction: #for every prediction, find the closest prototype and get the label
        all_distances_scipy = [1-scipy.spatial.distance.cosine(prediction, label) for label in hypersphere_vector] #1-distance gives similarity
        predicted_label.append(hypersphere_label[np.argmax(all_distances_scipy)])
    return predicted_label

