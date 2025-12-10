import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from project.scripts.model_init import EnhancementModel
from project.scripts.feature_encoder import FeatureEncoder
from project.scripts.multiscale_upsample import MultiScaleUpsampler

class MultiLevelFeatureFusion(nn.Module):
    def __init__(self, nf=64):
        """
        Fuses features from different scales:
        - Upsampled deep features (1024x1024)
        - Low-level skip connection (512x512 -> upsampled to 1024)
        - Mid-level skip connection (512x512 -> upsampled to 1024)
        """
        super(MultiLevelFeatureFusion, self).__init__()
        
        # Fusion convolution: Reduces concatenated channels back to nf
        # Inputs: Upsampled (nf) + Low Level (nf) + Mid Level (nf) = 3 * nf
        self.fusion_conv = nn.Conv2d(nf * 3, nf, 1, 1, 0, bias=True)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fusion_conv.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if self.fusion_conv.bias is not None:
            nn.init.constant_(self.fusion_conv.bias, 0)

    def forward(self, upsampled_feat, skip_features):
        """
        Args:
            upsampled_feat: (B, 64, 1024, 1024)
            skip_features: dict with 'low_level' and 'mid_level' (B, 64, 512, 512)
        Returns:
            out: (B, 64, 1024, 1024)
        """
        low_feat = skip_features['low_level']
        mid_feat = skip_features['mid_level']
        
        # Upsample skip features to match upsampled_feat size (1024x1024)
        target_size = upsampled_feat.shape[2:] # (1024, 1024)
        
        # Using bilinear interpolation for feature map resizing
        low_up = F.interpolate(low_feat, size=target_size, mode='bilinear', align_corners=False)
        mid_up = F.interpolate(mid_feat, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate along channel dimension
        concat_feat = torch.cat([upsampled_feat, low_up, mid_up], dim=1)
        
        # Fuse using 1x1 convolution
        out = self.fusion_conv(concat_feat)
        
        return out

def test_fusion():
    print("Testing Multi-Level Feature Fusion...")
    
    # 1. Setup Paths
    preprocessed_dir = Path("project/datasets/preprocessed")
    images = list(preprocessed_dir.glob("*.jpg")) + list(preprocessed_dir.glob("*.png"))
    
    if not images:
        print("X No preprocessed images found.")
        return

    test_img_path = images[0]
    print(f"Loading test image: {test_img_path.name}")

    # 2. Load and Prepare Image
    img = cv2.imread(str(test_img_path))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
    
    # 3. Initialize Pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nf = 64
    
    base_model = EnhancementModel(nf=nf, num_blocks=16, scale=2).to(device)
    encoder = FeatureEncoder(base_model).to(device)
    upsampler = MultiScaleUpsampler(base_model, nf=nf).to(device)
    fusion = MultiLevelFeatureFusion(nf=nf).to(device)
    
    img_tensor = img_tensor.to(device)

    # 4. Run Pipeline
    with torch.no_grad():
        # Encode
        encoded, skips = encoder(img_tensor)
        # Upsample
        upsampled = upsampler(encoded)
        # Fuse
        fused = fusion(upsampled, skips)
    
    # 5. Verify Outputs
    print(f"Fusion output shape: {fused.shape}")
    
    if fused.shape == (1, 64, 1024, 1024):
        print("✓ Shape verification passed")
    else:
        print(f"X Shape verification failed. Expected (1, 64, 1024, 1024), got {fused.shape}")

if __name__ == "__main__":
    test_fusion()
