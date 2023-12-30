import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    '''Squeeze-and-Excitation block.'''

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = F.relu(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out

class Block(nn.Module):
   """
   This class represents a block of operations in a CNN.
   Each block consists of three convolutional layers, optionally followed by a Squeeze-and-Excitation (SE) block, 
   and a shortcut connection that allows the input to be added to the output of the block.
   """

   def __init__(self, w_in, w_out, stride, group_width, bottleneck_ratio, se_ratio):
       """
       Initializes the block.
       
       Parameters:
       - w_in: number of input channels
       - w_out: number of output channels
       - stride: stride for the convolutional layers
       - group_width: width of the groups for the second convolutional layer
       - bottleneck_ratio: ratio for the bottleneck width
       - se_ratio: ratio for the Squeeze-and-Excitation block
       """
       super(Block, self).__init__()

       # 1x1 convolution
       w_b = int(round(w_out * bottleneck_ratio))
       self.conv1 = nn.Conv2d(w_in, w_b, kernel_size=1, bias=False)
       self.bn1 = nn.BatchNorm2d(w_b)

       # 3x3 convolution
       num_groups = w_b // group_width
       self.conv2 = nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_groups, bias=False)
       self.bn2 = nn.BatchNorm2d(w_b)

       # Squeeze-and-Excitation block
       self.with_se = se_ratio > 0
       if self.with_se:
           w_se = int(round(w_in * se_ratio))
           self.se = SE(w_b, w_se)

       # 1x1 convolution
       self.conv3 = nn.Conv2d(w_b, w_out, kernel_size=1, bias=False)
       self.bn3 = nn.BatchNorm2d(w_out)

       # Shortcut connection
       self.shortcut = nn.Sequential()
       if stride != 1 or w_in != w_out:
           self.shortcut = nn.Sequential(
               nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(w_out)
           )

   def forward(self, x):
       """
       Defines the forward pass of the block.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor
       """
       out = F.relu(self.bn1(self.conv1(x)))
       out = F.relu(self.bn2(self.conv2(out)))
       if self.with_se:
           out = self.se(out)
       out = self.bn3(self.conv3(out))
       out += self.shortcut(x)
       out = F.relu(out)
       return out



class RegNet(nn.Module):
   """
   This class represents a RegNet model.
   RegNet is a type of CNN architecture that uses a regularized convolutional layer structure.
   """

   def __init__(self, cfg, input_channels=3, num_classes=10):
       """
       Initializes the RegNet model.
       
       Parameters:
       - cfg: configuration dictionary containing parameters for the model
       - input_channels: number of input channels
       - num_classes: number of classes for classification
       """
       super(RegNet, self).__init__()
       self.cfg = cfg
       self.in_planes = 64
       self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(64)
       self.layer1 = self._make_layer(0)
       self.layer2 = self._make_layer(1)
       self.layer3 = self._make_layer(2)
       self.layer4 = self._make_layer(3)
       self.linear = nn.Linear(self.cfg['widths'][-1], num_classes)

   def _make_layer(self, idx):
       """
       Creates a layer of blocks.
       
       Parameters:
       - idx: index of the current layer
       
       Returns:
       - layers: a sequence of blocks
       """
       depth = self.cfg['depths'][idx]
       width = self.cfg['widths'][idx]
       stride = self.cfg['strides'][idx]
       group_width = self.cfg['group_width']
       bottleneck_ratio = self.cfg['bottleneck_ratio']
       se_ratio = self.cfg['se_ratio']

       layers = []
       for i in range(depth):
           s = stride if i == 0 else 1
           layers.append(Block(self.in_planes, width, s, group_width, bottleneck_ratio, se_ratio))
           self.in_planes = width
       return nn.Sequential(*layers)

   def forward(self, x):
       """
       Defines the forward pass of the model.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor
       """
       out = F.relu(self.bn1(self.conv1(x)))
       out = self.layer1(out)
       out = self.layer2(out)
       out = self.layer3(out)
       out = self.layer4(out)
       out = F.adaptive_avg_pool2d(out, (1, 1))
       out = out.view(out.size(0), -1)
       out = self.linear(out)
       return out



def RegNetX_200MF(input_channels=3, num_classes=10):
    cfg = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [1, 1, 2, 2],
        'group_width': 8,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg, input_channels=input_channels, num_classes=num_classes)


def RegNetX_400MF(input_channels=3, num_classes=10):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg, input_channels=input_channels, num_classes=num_classes)


def RegNetY_400MF(input_channels=3, num_classes=10):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0.25,
    }
    return RegNet(cfg, input_channels=input_channels, num_classes=num_classes)
