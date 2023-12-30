import torch
import torch.nn as nn


class VGG(nn.Module):
   """
   This class represents a VGG.
   VGG is a type of CNN architecture.
   It consists of several layers including convolutional layers, batch normalization layers, max pooling layers, and a fully connected layer.
   """

   def __init__(self, cfg, input_channels=1, num_classes=10): # Change input_channels to 1 for grayscale images
       """
       Initializes the VGG.
       
       Parameters:
       - cfg: configuration dictionary containing the number of output planes for each layer
       - input_channels: number of input channels (default is 1 for grayscale images)
       - num_classes: number of classes for classification (default is 10)
       """
       super(VGG, self).__init__()
       self.features = self._make_layers(cfg, in_channels=input_channels)
       self.classifier = nn.Linear(512, num_classes)

   def forward(self, x):
       """
       Defines the forward pass of the VGG.
       
       Parameters:
       - x: input tensor
       
       Returns:
       - out: output tensor after passing through the VGG
       """
       # Pass the input through the feature extraction layers
       out = self.features(x)

       # Reshape the output
       out = out.view(out.size(0), -1)

       # Pass the reshaped output through the classifier
       out = self.classifier(out)

       return out

   def _make_layers(self, cfg, in_channels=1): # Change in_channels to 1 for grayscale images
       """
       Creates the feature extraction layers of the VGG.
       
       Parameters:
       - cfg: configuration dictionary containing the number of output planes for each layer
       - in_channels: number of input channels (default is 1 for grayscale images)
       
       Returns:
       - layers: a sequence of layers
       """
       layers = []
       for x in cfg:
           if x == 'M':
               # Add a max pooling layer
               layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
           else:
               # Add a convolutional layer, a batch normalization layer, and a ReLU activation function
               layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                         nn.BatchNorm2d(x),
                         nn.ReLU(inplace=True)]
               in_channels = x
       # Add an average pooling layer at the end
       layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
       return nn.Sequential(*layers)



def VGG11(input_channels=3, num_classes=10):
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(cfg, input_channels=input_channels, num_classes=num_classes)

def VGG13(input_channels=3, num_classes=10):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(cfg, input_channels=input_channels, num_classes=num_classes)

def VGG16(input_channels=3, num_classes=10):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(cfg, input_channels=input_channels, num_classes=num_classes)

def VGG19(input_channels=3, num_classes=10):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return VGG(cfg, input_channels=input_channels, num_classes=num_classes)
