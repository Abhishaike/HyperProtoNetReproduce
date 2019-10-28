from hypersphere_opt import create_hypersphere_loss
from cifar_dataloader import get_cifar_data
from resnet_arch import resnet32
from train_and_test import train, test
import torch
import torch.optim as optim


output_dimension = 25
privileged_info = None
lr = .005
momentum = .001
use_cuda = True
batch_size = 128
epochs = 100

train_loader, test_loader = get_cifar_data(batch_size=batch_size)
unique_classes = train_loader.dataset.classes #privileged information, do not use for now
num_classes = len(unique_classes)
class_matched_points = create_hypersphere_loss(num_classes, output_dimension, unique_classes, use_privileged_info = False)

device = torch.device("cuda" if use_cuda else "cpu")

model = resnet32(output_dimension).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

if (args.save_model):
    torch.save(model.state_dict(), "mnist_cnn.pt")