from prototype_sphere_loss import regression_loss, classification_loss, joint_loss
import torch
import torchvision.transforms as transforms
from prototype_sphere_loss import classification_loss
import numpy as np
import scipy

def train(model, device, train_loader, optimizer, class_matched_points, epoch):
    model.train()
    for batch_idx, (local_batch, local_labels) in enumerate(train_loader):
        # Transfer to GPU
        hypersphere_vector = torch.FloatTensor([list(class_matched_points[class_num.item()]) for class_num in local_labels]) #get the vector of the label
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
            hypersphere_labels = assign_predicted_class(hypersphere_prediction.detach(), class_matched_points) #get closest matching prototypes
            correct = (np.array(hypersphere_labels) == np.array(local_labels.cpu())) * 1
            print('Train Accuracy: ', correct.mean())

        #cv2.imshow('hi',image[5].cpu().numpy().transpose(1, 2, 0)) #see example batch image


def test(model, device, test_loader, class_matched_points, epoch):
    model.eval()
    all_correct = []
    with torch.no_grad():
        for batch_idx, (local_batch, local_labels) in enumerate(test_loader):
            image, hypersphere_labels = local_batch.to(device), local_labels.to(device)
            hypersphere_prediction = model(image)
            pred_labels = assign_predicted_class(hypersphere_prediction, class_matched_points) #get closest matching prototypes
            correct = (np.array(hypersphere_labels.cpu()) == pred_labels)
            all_correct.extend(correct * 1)
    print('\n Epoch {0}, Test set accuracy: {1}'.format(epoch, np.array(all_correct).mean()))


def assign_predicted_class(hypersphere_prediction, class_matched_points):
    hypersphere_prediction = np.array(hypersphere_prediction.cpu())
    hypersphere_label, hypersphere_vector = list(class_matched_points.keys()), np.array(list(class_matched_points.values()))
    predicted_label = []
    for prediction in hypersphere_prediction: #for every prediction, find the closest prototype and get the label
        all_distances = [scipy.spatial.distance.cosine(prediction, label) for label in hypersphere_vector]
        predicted_label.append(hypersphere_label[np.argmin(all_distances)])
    return predicted_label
