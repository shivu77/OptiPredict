import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return identity + out

class EnhancementModel(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, nf=64, num_blocks=16, scale=2):
        super(EnhancementModel, self).__init__()
        self.scale = scale
        
        # 1. Input Conv
        self.conv_first = nn.Conv2d(num_in_ch, nf, 3, 1, 1, bias=True)
        
        # 2. Residual Blocks (Body)
        self.body = nn.Sequential(*[ResidualBlock(nf) for _ in range(num_blocks)])
        
        # 3. Post-Residual Conv (for global skip connection)
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # 4. Upsampling
        self.upconv = nn.Conv2d(nf, nf * scale * scale, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.act_up = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 5. Output Conv (HR)
        self.conv_last = nn.Conv2d(nf, num_out_ch, 3, 1, 1, bias=True)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat  # Global Skip Connection
        
        # Upsampling
        out = self.upconv(feat)
        out = self.pixel_shuffle(out)
        out = self.act_up(out)
        
        out = self.conv_last(out)
        return out

if __name__ == "__main__":
    # Define parameters
    SCALE = 2
    BLOCKS = 16
    FILTERS = 64
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancementModel(nf=FILTERS, num_blocks=BLOCKS, scale=SCALE).to(device)
    
    print(f"Model initialized: scale={SCALE}, residual_blocks={BLOCKS}, filters={FILTERS}")
    
    # Optional: Print basic verification of structure (commented out to keep output clean as requested)
    # print(model)
