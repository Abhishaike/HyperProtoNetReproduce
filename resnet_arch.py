import torch
import torch.nn as nn
import torchvision.models.resnet
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000, group_norm=False):
        if group_norm:
            norm_layer = lambda x: nn.GroupNorm(32, x)
        else:
            norm_layer = None
        super(ResNet, self).__init__(block, layers, num_classes, norm_layer=norm_layer)
        if not group_norm:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
            for i in range(2, 5):
                getattr(self, 'layer%d'%i)[0].conv1.stride = (2,2)
                getattr(self, 'layer%d'%i)[0].conv2.stride = (1,1)

def resnet32(num_outputs):
    """Constructs a ResNet-32 model. Number of outputs is hyperparameter that
    indicates the size of the vector to be compared to the hypersphere"""
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    model.fc = nn.Linear(512, num_outputs)
    return model