import torch
import torch.nn as nn

# https://ojs.aaai.org/index.php/AAAI/article/view/4851
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size = 3,
                 padding = 1,
                 bias = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, 
                                   in_channels, 
                                   kernel_size=kernel_size, 
                                   padding=padding,
                                   groups=in_channels,
                                   bias=bias)
        
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   bias=bias)
        
    def forward(self, x):
        x = self.depthwise()
        return self.pointwise(x)

# https://github.com/HansBambel/SmaAt-UNet/blob/snapshot-paper/models/unet_parts.py
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),      
        )
    
    def forward(self, x):
        return self.double_conv(x)
