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
        x = self.depthwise(x)
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
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
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
    
class SmaAtUNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.cbam = CBAM(out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv(x)
        skip = self.cbam(x)
        down = self.pool(x)
        return down, skip

class SmaAtUNetDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        assert x.shape[-2:] == skip.shape[-2:], (x.shape, skip.shape)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
        
class SmaAtUNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, base=64):
        super().__init__()
        self.out_channels = out_channels
        
        self.enc1 = SmaAtUNetEncoder(in_channels, base)
        self.enc2 = SmaAtUNetEncoder(base, base * 2)
        self.enc3 = SmaAtUNetEncoder(base * 2, base * 4)
        self.enc4 = SmaAtUNetEncoder(base * 4, base * 8)
        
        self.bottleneck_conv = DoubleConv(base * 8, base * 8)
        self.bottleneck_cbam = CBAM(base * 8)
        
        self.dec4 = SmaAtUNetDecoder(base * 8, base * 8, base * 4)
        self.dec3 = SmaAtUNetDecoder(base * 4, base * 4, base * 2)
        self.dec2 = SmaAtUNetDecoder(base * 2, base * 2, base)
        self.dec1 = SmaAtUNetDecoder(base, base, base)
        
        self.out = nn.Conv2d(base, out_channels, kernel_size=1)
        
    def forward(self, x):
        restore_sequence = False
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            if c != 1:
                raise ValueError(
                    f"SmaAtUNet expects single-channel frames when given 5D input, got C={c}"
                )
            x = x.reshape(b, t * c, h, w)
            restore_sequence = True
        elif x.ndim != 4:
            raise ValueError(f"SmaAtUNet expects 4D or 5D input, got shape {tuple(x.shape)}")

        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        x = self.bottleneck_conv(x)
        x = self.bottleneck_cbam(x)
        
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = self.out(x)
        if restore_sequence:
            x = x.unsqueeze(2)
        return x
