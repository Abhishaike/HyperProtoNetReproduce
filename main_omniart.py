from hypersphere_classes_opt_scipy import create_hypersphere_loss_wo_constraints, create_hypersphere_loss_w_constraints
from hypersphere_classes_opt_sgd import create_hypersphere_loss_w_sgd
from data_loaders import cifar_loader, imagenet_loader, mnist_loader, omniart_loader
from resnet_arch import resnet32
from train_and_test_omniart import train, test
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Polar Prototypical Regression")
    parser.add_argument("-o", dest="output_dimension", default=46, type=int)
    parser.add_argument("-p", dest="prototype_optimizer", default="sgd", type=str)
    parser.add_argument("-r", dest="model_optimizer", default="sgd", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.001, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-d", dest="decay", default=0.0001, type=float)
    parser.add_argument("-b", dest="batch_size", default=128, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("-c", dest="use_cuda", default=True, type=int)
    parser.add_argument("--seed", dest="seed", default=100, type=int)
    parser.add_argument("--operation", dest="task", default='joint', type=int) #could be classification, regression, or joint
    parser.add_argument("--taskweight", dest="taskweight", default=0.5, type=float) #weight of classification
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse user parameters and set device.
    args   = parse_args()

    output_dimension = args.output_dimension
    lr = args.learning_rate
    momentum  = args.momentum
    decay = args.decay
    batch_size = args.batch_size
    epochs = args.epochs
    device = torch.device("cuda" if args.use_cuda else "cpu")
    operation = args.operation
    task_weight = args.operation

    basedir = 'data/omniart/'
    trainfile = basedir + "train_complete.csv"
    testfile = basedir + "test_complete.csv"
    train_loader, test_loader = omniart_loader.load_omniart(basedir, trainfile, testfile, batch_size)

    unique_classes = list(set(train_loader.dataset.styles)) #class names privileged information, do not use for now
    unique_class_numbers = list(set(np.array(train_loader.dataset.stylelabels))) #
    num_classes = len(unique_classes)

    #use for style classification, choose prototype optimization based on argument
    if args.prototype_optimizer == 'sgd':
        classification_matched_points = create_hypersphere_loss_w_sgd(num_classes=num_classes,
                                                                      output_dimension=output_dimension,
                                                                      unique_class_numbers=unique_class_numbers)
    elif args.prototype_optimizer == 'slsqp':
        classification_matched_points = create_hypersphere_loss_w_constraints(num_classes = num_classes,
                                                                               output_dimension = output_dimension,
                                                                               unique_classes = unique_classes,
                                                                               unique_class_numbers = unique_class_numbers,
                                                                               use_privileged_info = False)
    elif args.prototype_optimizer == 'vfgs':
        classification_matched_points = create_hypersphere_loss_wo_constraints(num_classes = num_classes,
                                                                               output_dimension = output_dimension,
                                                                               unique_classes = unique_classes,
                                                                               unique_class_numbers = unique_class_numbers,
                                                                               use_privileged_info = False)

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
        train(model, device, train_loader, optimizer, epoch, classification_matched_points,
              operation, upper_bound_prototype, upper_bound, lower_bound, task_weight)
        test(model, device, test_loader, epoch, classification_matched_points,
              operation, upper_bound_prototype, upper_bound, lower_bound, task_weight)





