# Pytorch Reimplementation of the 2019 NeurIPS paper "Hyperspherical Prototype Networks"

## Introduction
This is a Python 3.6 reimplementation of the NeurIPS 2019 paper [‘Hyperspherical Prototype Networks’](https://arxiv.org/abs/1901.10514) in Pytorch. This paper proposes an extension to Prototype Networks, in which the prototypes are placed a priori with large margin separation, and remain unchanged during the training/testing process of the model. The paper suggests that this extension allows for more flexible classification, regression, and joint multi-task training of regression/classification, and with higher accuracy compared to typical Prototype Networks. 


## Benchmarking
The primary benchmarks of the paper are the following: 
1.	Benchmarking hypersphere prototype classification on CIFAR-100 with a variety of output-dimensions. 
2.	Benchmarking hypersphere prototype classification on ImageNet-200 with a variety of output-dimensions. 
3.	Benchmarking hypersphere prototype regression on OmniArt with one output dimension. 
4.	Benchmarking hypersphere prototype joint regression/classification on OmniArt with a variety of task weights. 
5.	Benchmarking hypersphere prototype classification on low-sample problems using CUB-200.
6.	Benchmarking hypersphere prototype classification on ImageNet-200 and CIFAR-100 using privileged information to place the prototypes.

This repository reimplements tasks 1-4, which are the main focuses of the paper. The paper tests (3) with 2 output dimensions as well, but as the results are near-identical to one another, we reimplemented just the one output dimension

However, the code is modular and amenable to both tasks 5 and 6; CUB-200 could easily be used within the current ‘train_and_test_imagenet_cifar.py’ file, and privileged information is included as a parameter/function in the prototype optimization files and only require the word2vec vectors and loss function to be supplied. 

## Results
1. CIFAR-100 : Classification

| Output Dimension | Original Acc.| Reimplementation Acc.|
| :---         |     :---:      |          ---: |
| 10   | 51.1     | 50.7    |
| 25     | 63.0       | TBD      |
| 50     | 64.7       | TBD      |
| 100     | 65.0       | TBD      |

1. ImageNet-200 : Classification

| Output Dimension | Original Acc.| Reimplementation Acc.|
| :---         |     :---:      |          ---: |
| 25   | 38.6     | TBD    |
| 50     | 44.7       | 40.1      |
| 100     | 44.6       | TBD      |
| 200     | 44.7       | TBD      |

1. OmniArt : Regression

| Learning Rate | Original Acc.| Reimplementation Acc.|
| :---         |     :---:      |          ---: |
| .01   | 84.4     |TBD    |
| .001   | 76.3     | 65.87    |

1. OmniArt : Joint Regression and Classification

| Task Weight | Original - Classification Acc. | Reimplementation - Classification Acc.| Original - Regression MAE| Reimplementation - Regression MAE|
| :---         |     :---:      |          ---: |     :---:      |          ---: |
| .01   | 46.6     | TBD    | 65.2     | TBD    |
| .1     | 51.2       | TBD      | 64.6       | TBD      |
| .25     | 54.5       | 53.1      | 64.1       | 69.71      |
| .5     | 52.6       | TBD      | 68.3       | TBD      |
| .9     | 51.4       | TBD      | 83.6       | TBD      |

## Downloading Data
For the OmniArt and ImageNet200 datasets, while in the home directory:
```
cd data/
wget -r -nH --cut-dirs=3 --no-parent --reject="index.html*" http://isis-data.science.uva.nl/mettes/hpn/data/imagenet200/
wget -r -nH --cut-dirs=3 --no-parent --reject="index.html*" http://isis-data.science.uva.nl/mettes/hpn/data/omniart/
```
The CIFAR-100 data should be downloaded automatically upon the appropriate CIFAR-100 train run. 

## Running train/test scripts
For the CIFAR-100 or ImageNet-200 (change dataset arguement as needed to 'imagenet', 'cifar' or 'mnist') classification tasks
```
python3 main_imagenet_cifar.py.py --dataset imagenet --seed 50
```

For the OmniArt regression (year prediction) task. Task weight should be 0.
```
python3 main_omniart.py --operation regression --seed 50 --taskweight 0
```

For the OmniArt regression (style prediction) task. Task weight should be 1.
```
python3 main_omniart.py --operation classification --seed 50 --taskweight 1
```

For the OmniArt joint (year + style prediction) task. Task weight should be between (0,1)
```
python3 main_omniart.py --operation joint --seed 50 --taskweight .25
```


## Extension
This repository also includes two alternative optimization process to the Stochastic Gradient Descent that the paper uses to place prototypes: Quasi-Newton BFGS, and Constrained Sequential Quadratic Programming with the constraint being that all produced vectors must be normed. For the most part, these alternate optimization did not change the results, and the training process defaults to SGD due to its speed. To make use of these optimization methods, change the -p flag to either 'slsqp' or 'bfgs'.

## Citations 
The original authors were extremely helpful in answering my questions, and their own implementation of the paper helped make some of the more confusing aspects of prototype regression much easier to understand: https://github.com/psmmettes/hpn. Code from this repository was used to format the Omniart dataset. 
```
@inproceedings{mettes2019hyperspherical,
  title={Hyperspherical Prototype Networks},
  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
Furthermore, the PyTorch implementation of the ResNet architecture was used: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


## Current Issues 
There isn't really a *bug* persay, but moreso just misleading: because of the normalization schemes for the training dataset, the mean absolute error for year predictions look extremely off on the training dataset. This doesn't affect the test groups which don't have the normalization applied to it, so the reported classification/style accuracies are correct, but just something to keep in mind. 


