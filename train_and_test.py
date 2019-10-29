from prototype_sphere_loss import regression_loss, classification_loss, joint_loss
import torch
import torchvision.transforms as transforms

def train(model, device, train_loader, optimizer, class_matched_points, epoch):
    model.train()
    for local_batch, local_labels in train_loader:
        # Transfer to GPU
        hypersphere_labels = torch.FloatTensor([list(class_matched_points[class_num.item()]) for class_num in local_labels])
        image, hypersphere_labels = local_batch.to(device), hypersphere_labels.to(device)
        optimizer.zero_grad()
        hypersphere_prediction = model(image)
        cosine_loss = (hypersphere_prediction, hypersphere_labels)
        cosine_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))