import torch.nn as nn
import torch.nn.functional as F

# Define the model
# Define the model
class Net(nn.Module):
  """
  Define a simple CNN with two convolutional layers and a fully connected layer.

  Attributes:
  conv1 (nn.Conv2d): First convolutional layer.
  conv2 (nn.Conv2d): Second convolutional layer.
  fc (nn.Linear): Fully connected layer.
  """

  def __init__(self, input_channels=3, num_classes=10):
    """
    Initialize the layers of the network.

    Parameters:
    input_channels (int, optional): Number of input channels. Defaults to 3.
    num_classes (int, optional): Number of output classes. Defaults to 10.
    """
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.fc = nn.Linear(32*8*8, num_classes)

  def forward(self, x):
    """
    Define the forward pass of the network.

    Parameters:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Output tensor.
    """
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 32*8*8)
    x = self.fc(x)
    return x



