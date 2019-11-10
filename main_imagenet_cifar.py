from hypersphere_classes_opt_scipy import create_hypersphere_loss_wo_constraints, create_hypersphere_loss_w_constraints
from hypersphere_classes_opt_sgd import create_hypersphere_loss_w_sgd
from data_loaders import cifar_loader, imagenet_loader, mnist_loader
from resnet_arch import resnet32
from train_and_test_imagenet_cifar import train, test
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Polar Prototypical Regression")
    parser.add_argument("-o", dest="output_dimension", default=46, type=int)
    parser.add_argument("-p", dest="prototype_optimizer", default="sgd", type=str)
    parser.add_argument("-r", dest="model_optimizer", default="sgd", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.01, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-d", dest="decay", default=0.0001, type=float)
    parser.add_argument("-b", dest="batch_size", default=128, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("-c", dest="use_cuda", default=True, type=bool)
    parser.add_argument("--seed", dest="seed", default=100, type=int)
    parser.add_argument("--dataset", dest="dataset", default='cifar', type=str) #could be classification, regression, or joint
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse user parameters
    args = parse_args()

    output_dimension = args.output_dimension
    lr = args.learning_rate
    momentum  = args.momentum
    decay = args.decay
    batch_size = args.batch_size
    epochs = args.epochs
    device = torch.device("cuda" if args.use_cuda else "cpu")
    dataset = args.datasetjn

    #choose dataset to be trained on
    if dataset == 'cifar':
        train_loader, test_loader = cifar_loader.get_cifar_data(batch_size=batch_size)
    elif dataset == 'imagenet':
        train_loader, test_loader = imagenet_loader.get_imagenet200_data(batch_size=batch_size)
    elif dataset == 'mnist':
        train_loader, test_loader = mnist_loader.get_mnist_data(batch_size=batch_size)

    unique_classes = list(set(train_loader.dataset.classes)) #class names privileged information, do not use for now
    unique_class_numbers = list(set(np.array(train_loader.dataset.targets))) #
    num_classes = len(unique_classes)

    #set prototypes. Output will be a dict of {class_label: prototype_vector}
    if args.prototype_optimizer == 'sgd':
        classification_matched_points = create_hypersphere_loss_w_sgd(num_classes=num_classes,
                                                                      output_dimension=output_dimension,
                                                                      unique_class_numbers=unique_class_numbers)
    elif args.prototype_optimizer == 'slsqp':
        classification_matched_points = create_hypersphere_loss_w_constraints(num_classes=num_classes,
                                                                              output_dimension=output_dimension,
                                                                              unique_classes=unique_classes,
                                                                              unique_class_numbers=unique_class_numbers,
                                                                              use_privileged_info=False)
    elif args.prototype_optimizer == 'vfgs':
        classification_matched_points = create_hypersphere_loss_wo_constraints(num_classes=num_classes,
                                                                               output_dimension=output_dimension,
                                                                               unique_classes=unique_classes,
                                                                               unique_class_numbers=unique_class_numbers,
                                                                               use_privileged_info=False)

    model = resnet32(output_dimension, dataset).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        if epoch%100 == 0: #at the 200th
            lr = lr/10
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
        train(model, device, train_loader, optimizer, epoch, classification_matched_points)
        test(model, device, test_loader, epoch, classification_matched_points)




