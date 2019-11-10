# Reimplementation of the 2019 NeurIPS paper "Hyperspherical Prototype Networks"

## Introduction
This is a reimplementation of the NeurIPS 2019 paper [‘Hyperspherical Prototype Networks’](https://arxiv.org/abs/1901.10514). This paper proposes an extension to Prototype Networks, in which the prototypes are placed a priori with large margin separation, and remain unchanged during the training/testing process of the model. This paper suggests  that this extension allows for more flexible classification, regression, and joint multi-task training of regression/classification, and with higher accuracy compared to typical Prototype Networks


## Benchmarking
The primary benchmarks of the paper are the following: 
1.	Benchmarking hypersphere prototype classification on ImageNet-200 with a variety of output-dimensions. 
2.	Benchmarking hypersphere prototype classification on CIFAR-100 with a variety of output-dimensions. 
3.	Benchmarking hypersphere prototype regression on OmniArt with one output dimension. 
4.	Benchmarking hypersphere prototype joint regression/classification on OmniArt with a variety of task weights. 
5.	Benchmarking hypersphere prototype classification on low-sample problems using CUB-200.
6.	Benchmarking hypersphere prototype classification on ImageNet-200 and CIFAR-100 using privileged information to place the prototypes.

This repository reimplements tasks 1-4, which are the main focuses of the paper. 

However, the code is modular and amenable to both tasks 5 and 6; CUB-200 could easily be used within the current ‘train_and_test_imagenet_cifar.py’ file, and privileged information is included as a parameter/function in the prototype optimization files and only require the word2vec vectors and loss function to be supplied. 

## Results

## Extension
This repository also includes two alternative optimization process to the Stochastic Gradient Descent that the paper used to place prototypes: Quasi-Newton BFGS, and Constrained Sequential Quadratic Programming with the constraint being that all produced vectors must be normed. For the most part, these alternate optimization did not change the results, and the training process defaults to SGD due to its speed. 

## Citations 
The original authors were extremely helpful, and their own implementation of the paper helped make some of the more confusing aspects of prototype regression much easier to understand: https://github.com/psmmettes/hpn. Code from this repository was used to format the Omniart dataset. 
```
@inproceedings{mettes2019hyperspherical,
  title={Hyperspherical Prototype Networks},
  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
Furthermore, the PyTorch implementation of the ResNet architecture was used: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

