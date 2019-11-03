from prototype_sphere_loss import regression_loss, classification_loss, joint_loss
import torch
import torchvision.transforms as transforms
from prototype_sphere_loss import classification_loss
import numpy as np

def train(model, device, train_loader, optimizer, class_matched_points, epoch):
    model.train()
    for batch_idx, (local_batch, local_labels) in enumerate(train_loader):
        # Transfer to GPU
        hypersphere_labels = torch.FloatTensor([list(class_matched_points[class_num.item()]) for class_num in local_labels])
        image, hypersphere_labels = local_batch.to(device), hypersphere_labels.to(device)
        optimizer.zero_grad()
        hypersphere_prediction = model(image)
        cosine_loss = classification_loss(hypersphere_prediction, hypersphere_labels)
        cosine_loss.backward()
        optimizer.step()
        if batch_idx%16 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cosine_loss.item()))
        #cv2.imshow('hi',image[5].cpu().numpy().transpose(1, 2, 0)) #see example batch image


def test(model, device, test_loader, class_matched_points, epoch):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    with torch.no_grad():
        for batch_idx, (local_batch, local_labels) in enumerate(test_loader):
            image, class_labels = local_batch.to(device), local_labels.to(device)
            hypersphere_prediction = model(image)
            pred_labels = assign_predicted_class(device, hypersphere_prediction, class_matched_points) #get closest matching prototypes
            corrects = (np.array(class_labels.cpu()) == pred_labels)

    print('\n Epoch {0}, Test set accuracy: {1}'.format(epoch, corrects))


def assign_predicted_class(device, hypersphere_prediction, class_matched_points):
    label_target = []
    for prediction in hypersphere_prediction: #for every prediction
        max_cosine_similarity = 0 #reset similarity
        temp_pred_target = 0
        for label, label_prototype in class_matched_points.items(): #find the closest prototype according to cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(torch.FloatTensor(label_prototype).to(device), prediction, dim=0, eps=1e-8)
            if cosine_similarity > max_cosine_similarity: #if the similarity is higher than last highest, save it
                max_cosine_similarity = cosine_similarity
                temp_pred_target = label
        label_target.append(temp_pred_target) #save the target with the highest cosine similarity
    return label_target
