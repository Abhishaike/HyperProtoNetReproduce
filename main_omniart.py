from hypersphere_classes_opt_scipy import create_hypersphere_loss_wo_constraints, create_hypersphere_loss_w_constraints
from hypersphere_classes_opt_sgd import create_hypersphere_loss_w_sgd
from data_loaders import cifar_loader, imagenet_loader, mnist_loader, omniart_loader
from resnet_arch import resnet32
from train_and_test_omniart import train, test
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

output_dimension = 10
privileged_info = None
lr = .01
momentum = .9
use_cuda = True
batch_size = 12
epochs = 300
device = torch.device("cuda" if use_cuda else "cpu")
operation = 'classification'
class_weight = .5

basedir = 'data/omniart/'
trainfile = basedir + "train_complete.csv"
testfile = basedir + "test_complete.csv"

train_loader, test_loader = omniart_loader.load_omniart(basedir, trainfile, testfile, batch_size)

unique_classes = list(set(train_loader.dataset.styles)) #class names privileged information, do not use for now
unique_class_numbers = list(set(np.array(train_loader.dataset.stylelabels))) #
num_classes = len(unique_classes)

#use for style classification
# classification_matched_points = create_hypersphere_loss_w_constraints(num_classes = num_classes,
#                                                                        output_dimension = output_dimension,
#                                                                        unique_classes = unique_classes,
#                                                                        unique_class_numbers = unique_class_numbers,
#                                                                        use_privileged_info = False)
classification_matched_points = create_hypersphere_loss_w_sgd(num_classes = num_classes,
                                                     output_dimension = output_dimension,
                                                     unique_class_numbers = unique_class_numbers)

#use for year regression
years = train_loader.dataset.years


# upper bound prototype, corresponds to p_u in equation 7
upper_bound_prototype = np.zeros(output_dimension)
upper_bound_prototype[0] = 1

#ground truth regression value, corresponds to r_i in equation 7
lower_bound, upper_bound = min(years), max(years) #get lower and upper bound
normalized_years = 2. * ((years - lower_bound) / (upper_bound - lower_bound)) - 1 #normalize the years between the lower and upper bound to 0/1
train_loader.dataset.years = normalized_years


model = resnet32(output_dimension, dataset='omniart').to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)


for epoch in range(1, epochs + 1):
    if epoch%100 == 0:
        lr = lr/10
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    train(model, device, train_loader, optimizer, epoch, classification_matched_points, operation, upper_bound_prototype, class_weight)
    test(model, device, test_loader, classification_matched_points, epoch, operation, upper_bound, lower_bound)




