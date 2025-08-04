import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ColorEmbedding(nn.Module):
    """Embedding layer for color conditioning"""
    def __init__(self, num_colors, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_colors, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, color_idx, height, width):
        # color_idx: (batch_size,)
        # Returns: (batch_size, embed_dim, height, width)
        embed = self.embedding(color_idx)  # (batch_size, embed_dim)
        embed = embed.unsqueeze(-1).unsqueeze(-1)  # (batch_size, embed_dim, 1, 1)
        embed = embed.expand(-1, -1, height, width)  # (batch_size, embed_dim, height, width)
        return embed

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, num_colors=9, bilinear=False):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Color embedding
        self.color_embedding = ColorEmbedding(num_colors, embed_dim=64)
        
        # Initial convolution (RGB + color embedding)
        self.inc = DoubleConv(n_channels + 64, 64)
        
        # Encoder
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, n_classes)

    def forward(self, x, color_idx):
        # x: (batch_size, 3, H, W)
        # color_idx: (batch_size,)
        
        batch_size, _, height, width = x.shape
        
        # Get color embedding
        color_embed = self.color_embedding(color_idx, height, width)
        
        # Concatenate image and color embedding
        x_with_color = torch.cat([x, color_embed], dim=1)
        
        # UNet forward pass
        x1 = self.inc(x_with_color)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)  # Output in [0, 1] range

# Color mapping
COLOR_MAP = {
    'red': 0, 'green': 1, 'blue': 2, 'yellow': 3, 
    'orange': 4, 'purple': 5, 'cyan': 6, 'magenta': 7
}

def get_color_idx(color_name):
    """Convert color name to index"""
    return COLOR_MAP.get(color_name.lower(), 0)


if __name__ == "__main__":
    model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=len(COLOR_MAP))
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    colors = torch.tensor([0, 1])  # red, green
    
    output = model(x, colors)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
