import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
   """
   This class defines the Inception module used in the GoogleNet architecture.
   
   Attributes:
   b1, b2, b3, b4: These are the four branches of the Inception module. Each branch consists of several layers including convolution, batch normalization, and ReLU activation.
   """

   def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        """
        Initializes the Inception module.

        Parameters:
        in_planes (int): The number of input planes.
        n1x1, n3x3red, n5x5red, pool_planes (int): The number of filters in the 1x1 convolution, the 3x3 convolution after reduction, the 5x5 convolution after reduction, and the max pooling respectively.
        n3x3, n5x5 (int): The number of filters in the 3x3 convolution and the 5x5 convolution respectively.
        """
        super(Inception, self).__init__()

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

   def forward(self, x):
        """
        Defines the forward pass of the Inception module.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor.
        """
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
   """
   This class defines the GoogLeNet architecture.
   
   Attributes:
   pre_layers: This is the initial layer of the network which includes a convolution, batch normalization, and ReLU activation.
   a3, b3, a4, b4, c4, d4, e4, a5, b5: These are the inception modules of the network. Each inception module consists of several layers including convolution, batch normalization, and ReLU activation.
   maxpool: This is the max pooling layer of the network.
   avgpool: This is the average pooling layer of the network.
   linear: This is the final fully connected layer of the network.
   """

   def __init__(self, input_channels=3, num_classes=10): # Change input_channels to 1 for grayscale images
       """
       Initializes the GoogLeNet architecture.
       
       Parameters:
       input_channels (int): The number of input channels. Default is 1 for grayscale images.
       num_classes (int): The number of classes for classification. Default is 10.
       """
       super(GoogLeNet, self).__init__()
       self.pre_layers = nn.Sequential(
           nn.Conv2d(input_channels, 192, kernel_size=3, padding=1),
           nn.BatchNorm2d(192),
           nn.ReLU(True),
       )

       self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
       self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

       self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

       self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
       self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
       self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
       self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
       self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

       self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
       self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

       self.avgpool = nn.AvgPool2d(8, stride=1)
       self.linear = nn.Linear(1024, num_classes)

   def forward(self, x):
       """
       Defines the forward pass of the GoogLeNet architecture.
       
       Parameters:
       x (torch.Tensor): The input tensor.
       
       Returns:
       torch.Tensor: The output tensor.
       """
       out = self.pre_layers(x)
       out = self.a3(out)
       out = self.b3(out)
       out = self.maxpool(out)
       out = self.a4(out)
       out = self.b4(out)
       out = self.c4(out)
       out = self.d4(out)
       out = self.e4(out)
       out = self.maxpool(out)
       out = self.a5(out)
       out = self.b5(out)
       out = self.avgpool(out)
       out = out.view(out.size(0), -1)
       out = self.linear(out)
       return out
