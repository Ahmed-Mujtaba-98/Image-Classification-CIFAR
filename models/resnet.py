import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
   """
   This class represents a basic block of a ResNet model.
   ResNet is a type of CNN architecture that uses residual connections to mitigate the problem of vanishing gradients.
   """

   # Expansion factor for the number of filters in the convolutional layers
   expansion = 1

   def __init__(self, in_planes, planes, stride=1):
       """
       Initializes the BasicBlock.
       
       Parameters:
       - in_planes: number of input planes
       - planes: number of output planes
       - stride: stride for the convolutional layers
       """
       super(BasicBlock, self).__init__()

       # First convolutional layer
       self.conv1 = nn.Conv2d(
           in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(planes)

       # Second convolutional layer
       self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                             stride=1, padding=1, bias=False)
       self.bn2 = nn.BatchNorm2d(planes)

       # Shortcut connection
       self.shortcut = nn.Sequential()
       if stride != 1 or in_planes != self.expansion*planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(self.expansion*planes)
           )

   def forward(self, x):
       """
       Defines the forward pass of the BasicBlock.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor
       """
       # Apply first convolutional layer and batch normalization
       out = F.relu(self.bn1(self.conv1(x)))

       # Apply second convolutional layer and batch normalization
       out = self.bn2(self.conv2(out))

       # Add shortcut connection
       out += self.shortcut(x)

       # Apply ReLU activation function
       out = F.relu(out)

       return out

class Bottleneck(nn.Module):
   """
   This class represents a bottleneck block of a ResNet model.
   ResNet is a type of CNN architecture that uses residual connections to mitigate the problem of vanishing gradients.
   """

   # Expansion factor for the number of filters in the convolutional layers
   expansion = 4

   def __init__(self, in_planes, planes, stride=1):
       """
       Initializes the Bottleneck.
       
       Parameters:
       - in_planes: number of input planes
       - planes: number of output planes
       - stride: stride for the convolutional layers
       """
       super(Bottleneck, self).__init__()

       # First convolutional layer
       self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
       self.bn1 = nn.BatchNorm2d(planes)

       # Second convolutional layer
       self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                             stride=stride, padding=1, bias=False)
       self.bn2 = nn.BatchNorm2d(planes)

       # Third convolutional layer
       self.conv3 = nn.Conv2d(planes, self.expansion *
                             planes, kernel_size=1, bias=False)
       self.bn3 = nn.BatchNorm2d(self.expansion*planes)

       # Shortcut connection
       self.shortcut = nn.Sequential()
       if stride != 1 or in_planes != self.expansion*planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(self.expansion*planes)
           )

   def forward(self, x):
       """
       Defines the forward pass of the Bottleneck.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor
       """
       # Apply first convolutional layer and batch normalization
       out = F.relu(self.bn1(self.conv1(x)))

       # Apply second convolutional layer and batch normalization
       out = F.relu(self.bn2(self.conv2(out)))

       # Apply third convolutional layer and batch normalization
       out = self.bn3(self.conv3(out))

       # Add shortcut connection
       out += self.shortcut(x)

       # Apply ReLU activation function
       out = F.relu(out)

       return out

class ResNet(nn.Module):
   """
   This class represents a ResNet model.
   ResNet is a type of CNN architecture that uses residual connections to mitigate the problem of vanishing gradients.
   """

   def __init__(self, block, num_blocks, input_channels=3, num_classes=10):
       """
       Initializes the ResNet.
       
       Parameters:
       - block: type of block to use in the ResNet (e.g., BasicBlock, Bottleneck)
       - num_blocks: number of blocks in each layer
       - input_channels: number of input channels
       - num_classes: number of classes for classification
       """
       super(ResNet, self).__init__()
       self.in_planes = 64

       # First convolutional layer
       self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3,
                             stride=1, padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(64)

       # Layers
       self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
       self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
       self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
       self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

       # Linear layer for final classification
       self.linear = nn.Linear(512*block.expansion, num_classes)

   def _make_layer(self, block, planes, num_blocks, stride):
       """
       Creates a layer of blocks.
       
       Parameters:
       - block: type of block to use in the layer
       - planes: number of output planes
       - num_blocks: number of blocks in the layer
       - stride: stride for the convolutional layers
       
       Returns:
       - layer: a sequence of blocks
       """
       strides = [stride] + [1]*(num_blocks-1)
       layers = []
       for stride in strides:
           layers.append(block(self.in_planes, planes, stride))
           self.in_planes = planes * block.expansion
       return nn.Sequential(*layers)

   def forward(self, x):
       """
       Defines the forward pass of the ResNet.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor
       """
       # Apply first convolutional layer and batch normalization
       out = F.relu(self.bn1(self.conv1(x)))

       # Apply layers
       out = self.layer1(out)
       out = self.layer2(out)
       out = self.layer3(out)
       out = self.layer4(out)

       # Apply average pooling
       out = F.avg_pool2d(out, 4)

       # Flatten the tensor
       out = out.view(out.size(0), -1)

       # Apply linear layer for final classification
       out = self.linear(out)

       return out



def ResNet18(input_channels=3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_channels=input_channels, num_classes=num_classes)


def ResNet34(input_channels=3, num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], input_channels=input_channels, num_classes=num_classes)


def ResNet50(input_channels=3, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], input_channels=input_channels, num_classes=num_classes)


def ResNet101(input_channels=3, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], input_channels=input_channels, num_classes=num_classes)


def ResNet152(input_channels=3, num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], input_channels=input_channels, num_classes=num_classes)
