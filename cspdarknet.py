import torch
import torch.nn as nn
import math

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(CSPLayer, self).__init__()
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)
        self.split_conv0 = BasicConv(out_channels, out_channels, 1)
        self.split_conv1 = BasicConv(out_channels, out_channels, 1)
        self.blocks_conv = nn.Sequential(
            *[Resblock(out_channels, out_channels) for _ in range(num_blocks)],
            BasicConv(out_channels, out_channels, 1)
        )
        self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x

class CSPDarknet(nn.Module):
    def __init__(self, base_channels=32, base_depth=1):
        super(CSPDarknet, self).__init__()
        # 初始卷积层
        self.conv1 = BasicConv(3, base_channels, 3, stride=1)
        self.conv2 = BasicConv(base_channels, base_channels*2, 3, stride=2)
        self.conv3 = BasicConv(base_channels*2, base_channels*2, 1)
        
        # CSP层
        self.stage1 = CSPLayer(base_channels*2, base_channels*4, base_depth)
        self.stage2 = CSPLayer(base_channels*4, base_channels*8, base_depth)
        self.stage3 = CSPLayer(base_channels*8, base_channels*16, base_depth)
        self.stage4 = CSPLayer(base_channels*16, base_channels*32, base_depth)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        
        return [feat1, feat2, feat3]

def cspdarknet53(pretrained=False):
    model = CSPDarknet(base_channels=32, base_depth=1)
    if pretrained:
        # 这里可以添加预训练权重的加载逻辑
        pass
    return model 