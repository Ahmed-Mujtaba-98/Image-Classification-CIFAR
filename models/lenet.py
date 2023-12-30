import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
   """
   This class defines the LeNet architecture.
   
   Attributes:
   conv1, conv2: These are the convolutional layers of the network. Each convolutional layer includes a convolution operation followed by a ReLU activation.
   fc1, fc2, fc3: These are the fully connected layers of the network. Each fully connected layer includes a linear transformation followed by a ReLU activation.
   """

   def __init__(self, input_channels=3, num_classes=10):
       """
       Initializes the LeNet architecture.
       
       Parameters:
       input_channels (int): The number of input channels. Default is 3 for RGB images.
       num_classes (int): The number of classes for classification. Default is 10.
       """
       super(LeNet, self).__init__()
       self.conv1 = nn.Conv2d(input_channels, 6, 5)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16 * 5 * 5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, num_classes)

   def forward(self, x):
       """
       Defines the forward pass of the LeNet architecture.
       
       Parameters:
       x (torch.Tensor): The input tensor.
       
       Returns:
       torch.Tensor: The output tensor.
       """
       out = F.relu(self.conv1(x))
       out = F.max_pool2d(out, 2)
       out = F.relu(self.conv2(out))
       out = F.max_pool2d(out, 2)
       out = out.view(out.size(0), -1)
       out = F.relu(self.fc1(out))
       out = F.relu(self.fc2(out))
       out = self.fc3(out)
       return out

