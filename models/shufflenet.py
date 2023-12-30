import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):
   """
   This class represents a ShuffleBlock.
   ShuffleBlock is a type of operation used in ShuffleNet, a type of CNN architecture.
   It rearranges the channels of the input tensor in a specific way.
   """

   def __init__(self, groups):
       """
       Initializes the ShuffleBlock.
       
       Parameters:
       - groups: number of groups to divide the channels into
       """
       super(ShuffleBlock, self).__init__()
       self.groups = groups

   def forward(self, x):
       """
       Defines the forward pass of the ShuffleBlock.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor after channel shuffling
       """
       # Get the dimensions of the input tensor
       N, C, H, W = x.size()

       # Get the number of groups
       g = self.groups

       # Rearrange the channels of the input tensor
       # [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
       out = x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

       return out


class Bottleneck(nn.Module):
   """
   This class represents a Bottleneck layer.
   Bottleneck is a type of operation used in ResNet, a type of CNN architecture.
   It consists of three convolutional layers and a shortcut connection.
   """

   def __init__(self, in_planes, out_planes, stride, groups):
       """
       Initializes the Bottleneck layer.
       
       Parameters:
       - in_planes: number of input planes
       - out_planes: number of output planes
       - stride: stride of the convolution
       - groups: number of groups to divide the channels into
       """
       super(Bottleneck, self).__init__()
       self.stride = stride

       # Calculate the number of middle planes
       mid_planes = out_planes // 4

       # Determine the number of groups for the first convolution
       g = 1 if in_planes == 24 else groups

       # Define the first convolutional layer
       self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)

       # Define the batch normalization layer for the first convolution
       self.bn1 = nn.BatchNorm2d(mid_planes)

       # Define the shuffle block for the first convolution
       self.shuffle1 = ShuffleBlock(groups=g)

       # Define the second convolutional layer
       self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)

       # Define the batch normalization layer for the second convolution
       self.bn2 = nn.BatchNorm2d(mid_planes)

       # Define the third convolutional layer
       self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)

       # Define the batch normalization layer for the third convolution
       self.bn3 = nn.BatchNorm2d(out_planes)

       # Define the shortcut connection
       self.shortcut = nn.Sequential()
       if stride == 2:
           self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

   def forward(self, x):
       """
       Defines the forward pass of the Bottleneck layer.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor after passing through the bottleneck layer
       """
       # Pass the input through the first convolution, batch normalization, and shuffle block
       out = F.relu(self.bn1(self.conv1(x)))

       # Apply the shuffle block to the output
       out = self.shuffle1(out)

       # Pass the output through the second convolution and batch normalization
       out = F.relu(self.bn2(self.conv2(out)))

       # Pass the output through the third convolution and batch normalization
       out = self.bn3(self.conv3(out))

       # Create the shortcut connection
       res = self.shortcut(x)

       # Combine the output of the bottleneck layer and the shortcut connection
       out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)

       return out



class ShuffleNet(nn.Module):
   """
   This class represents a ShuffleNet.
   ShuffleNet is a type of CNN architecture.
   It consists of several layers including convolutional layers, batch normalization layers, and linear layers.
   """

   def __init__(self, cfg, input_channels=3, num_classes=10):
       """
       Initializes the ShuffleNet.
       
       Parameters:
       - cfg: configuration dictionary containing the number of output planes, number of blocks, and number of groups
       - input_channels: number of input channels (default is 3 for RGB images)
       - num_classes: number of classes for classification (default is 10)
       """
       super(ShuffleNet, self).__init__()
       out_planes = cfg['out_planes']
       num_blocks = cfg['num_blocks']
       groups = cfg['groups']

       # Define the first convolutional layer
       self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=1, bias=False)

       # Define the batch normalization layer for the first convolution
       self.bn1 = nn.BatchNorm2d(24)

       # Initialize the number of input planes
       self.in_planes = 24

       # Define the first layer
       self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)

       # Define the second layer
       self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)

       # Define the third layer
       self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)

       # Define the final linear layer
       self.linear = nn.Linear(out_planes[2], num_classes)

   def _make_layer(self, out_planes, num_blocks, groups):
       """
       Creates a layer of Bottleneck blocks.
       
       Parameters:
       - out_planes: number of output planes
       - num_blocks: number of blocks in the layer
       - groups: number of groups to divide the channels into
       
       Returns:
       - layers: a sequence of Bottleneck blocks
       """
       layers = []
       for i in range(num_blocks):
           stride = 2 if i == 0 else 1
           cat_planes = self.in_planes if i == 0 else 0
           layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
           self.in_planes = out_planes
       return nn.Sequential(*layers)

   def forward(self, x):
       """
       Defines the forward pass of the ShuffleNet.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor after passing through the ShuffleNet
       """
       # Pass the input through the first convolution and batch normalization
       out = F.relu(self.bn1(self.conv1(x)))

       # Pass the output through the first layer
       out = self.layer1(out)

       # Pass the output through the second layer
       out = self.layer2(out)

       # Pass the output through the third layer
       out = self.layer3(out)

       # Apply average pooling to the output
       out = F.avg_pool2d(out, 4)

       # Flatten the output
       out = out.view(out.size(0), -1)

       # Pass the flattened output through the final linear layer
       out = self.linear(out)

       return out



def ShuffleNetG2(input_channels=3, num_classes=10):
    cfg = {
        'out_planes': [200,400,800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
    return ShuffleNet(cfg, input_channels=input_channels, num_classes=num_classes)

def ShuffleNetG3(input_channels=3, num_classes=10):
    cfg = {
        'out_planes': [240,480,960],
        'num_blocks': [4,8,4],
        'groups': 3
    }
    return ShuffleNet(cfg, input_channels=input_channels, num_classes=num_classes)
