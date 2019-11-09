from hypersphere_classes_opt_scipy import create_hypersphere_loss_wo_constraints, create_hypersphere_loss_w_constraints
from hypersphere_classes_opt_sgd import create_hypersphere_loss_w_sgd
from data_loaders import cifar_loader, imagenet_loader, mnist_loader
from resnet_arch import resnet32
from train_and_test_imagenet_cifar import train, test
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

output_dimension = 25
privileged_info = None
lr = .01
momentum = .9
use_cuda = True
batch_size = 128
epochs = 300
device = torch.device("cuda" if use_cuda else "cpu")
dataset = 'imagenet'

if dataset == 'cifar':
    train_loader, test_loader = cifar_loader.get_cifar_data(batch_size=batch_size)
elif dataset == 'imagenet':
    train_loader, test_loader = imagenet_loader.get_imagenet200_data(batch_size=batch_size)

unique_classes = list(set(train_loader.dataset.classes)) #class names privileged information, do not use for now
unique_class_numbers = list(set(np.array(train_loader.dataset.targets))) #
num_classes = len(unique_classes)


# classification_matched_points = create_hypersphere_loss_w_constraints(num_classes = num_classes,
#                                                                        output_dimension = output_dimension,
#                                                                        unique_classes = unique_classes,
#                                                                        unique_class_numbers = unique_class_numbers,
#                                                                        use_privileged_info = False)
classification_matched_points = create_hypersphere_loss_w_sgd(num_classes = num_classes,
                                                     output_dimension = output_dimension,
                                                     unique_class_numbers = unique_class_numbers)

model = resnet32(output_dimension, dataset).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)

# plt.scatter(np.array(list(class_matched_points.values()))[:,0], np.array(list(class_matched_points.values()))[:,1])
# plt.scatter(init_hyperspheres[:,0], init_hyperspheres[:,1])

for epoch in range(1, epochs + 1):
    if epoch%100 == 0:
        lr = lr/10
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    train(model, device, train_loader, optimizer, epoch, classification_matched_points)
    test(model, device, test_loader, epoch, classification_matched_points)




