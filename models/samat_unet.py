import torch
import torch.nn as nn
import torch.nn.functional as F

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

# https://arxiv.org/pdf/1807.06521
# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid_channels = max(channels // reduction_ratio, 1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False)            
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        attention = torch.sigmoid(x)
        return x * attention
        

class CBAM(nn.Module):
    def __init__(self, channels, channel_reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, 
                                                  reduction_ratio=channel_reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
