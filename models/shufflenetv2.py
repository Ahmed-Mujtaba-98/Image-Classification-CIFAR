import torch
import torch.nn as nn
import torch.nn.functional as F

configs = {
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        """
        Initializes a ShuffleBlock.

        Parameters:
            - groups (int): Number of groups for channel shuffling.
        """
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """
        Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
        """
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class SplitBlock(nn.Module):
    def __init__(self, ratio):
        """
        Initializes a SplitBlock.

        Parameters:
            - ratio (float): Ratio for splitting the input tensor.
        """
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        # Split the input tensor along the channel dimension
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]

class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        """
        Initializes a BasicBlock.

        Parameters:
            - in_channels (int): Number of input channels.
            - split_ratio (float): Ratio for splitting the input tensor.
        """
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        # 1x1 pointwise convolution
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # 3x3 depthwise convolution
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # 1x1 pointwise convolution
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        # First pointwise convolution followed by batch normalization and ReLU
        out = F.relu(self.bn1(self.conv1(x2)))
        # Depthwise convolution followed by batch normalization
        out = self.bn2(self.conv2(out))
        # Second pointwise convolution followed by batch normalization
        out = F.relu(self.bn3(self.conv3(out)))
        # Concatenate the split tensor and apply channel shuffle
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes a DownBlock.

        Parameters:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.
        """
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # Left branch
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # Right branch
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # Left branch
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # Right branch
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # Concatenate the two branches and apply channel shuffle
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, net_size, input_channels=3, num_classes=10):
        """
        Initializes a ShuffleNetV2 model.

        Parameters:
            - net_size (str): Size of the ShuffleNetV2 model (e.g., '1.5x', '2.0x').
        """
        super(ShuffleNetV2, self).__init__()
        # Get configurations for the network size
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']
        # Initial 3x3 convolution layer
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        # Construct the layers with DownBlocks and BasicBlocks
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        # Final 1x1 convolution layer
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        # Fully connected layer for classification
        self.linear = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial 3x3 convolution followed by batch normalization and ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply each layer sequentially
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # Final 1x1 convolution followed by batch normalization and ReLU
        out = F.relu(self.bn2(self.conv2(out)))
        # Global average pooling
        out = F.avg_pool2d(out, 4)
        # Flatten and pass through the fully connected layer
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

