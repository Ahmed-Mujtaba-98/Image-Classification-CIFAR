import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
   """
   MobileNet class for creating a MobileNet model.
   
   Attributes:
       cfg (list): Configuration for the MobileNet architecture. Each element in the list represents a layer in the network.
                  An integer represents the number of output channels for a convolutional layer, while a tuple (e.g., (128,2)) 
                  represents a convolutional layer with 128 output channels and a stride of 2.
   """
   cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

   def __init__(self, input_channels=3, num_classes=10):
       """
       Initializes the MobileNet model.
       
       Args:
           input_channels (int, optional): Number of input channels. Defaults to 3.
           num_classes (int, optional): Number of output classes. Defaults to 10.
       """
       super(MobileNet, self).__init__()
       self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(32)
       self.layers = self._make_layers(in_planes=32)
       self.linear = nn.Linear(1024, num_classes)

   def _make_layers(self, in_planes):
       """
       Creates the layers of the MobileNet model based on the configuration.
       
       Args:
           in_planes (int): Number of input channels.
           
       Returns:
           nn.Sequential: Sequential container of layers.
       """
       layers = []
       for x in self.cfg:
           out_planes = x if isinstance(x, int) else x[0]
           stride = 1 if isinstance(x, int) else x[1]
           layers.append(Block(in_planes, out_planes, stride))
           in_planes = out_planes
       return nn.Sequential(*layers)

   def forward(self, x):
       """
       Defines the forward pass of the MobileNet model.
       
       Args:
           x (torch.Tensor): Input tensor.
           
       Returns:
           torch.Tensor: Output tensor.
       """
       out = F.relu(self.bn1(self.conv1(x)))
       out = self.layers(out)
       out = F.avg_pool2d(out, 2)
       out = out.view(out.size(0), -1)
       out = self.linear(out)
       return out
