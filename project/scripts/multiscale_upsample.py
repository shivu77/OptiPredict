import torch
import torch.nn as nn
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from project.scripts.model_init import EnhancementModel
from project.scripts.feature_encoder import FeatureEncoder

class MultiScaleUpsampler(nn.Module):
    def __init__(self, model, nf=64):
        """
        Handles the upsampling stage of the pipeline.
        Extends the base model with refinement layers.
        """
        super(MultiScaleUpsampler, self).__init__()
        # Use upsampling layers from the base model
        self.upconv = model.upconv
        self.pixel_shuffle = model.pixel_shuffle
        self.act_up = model.act_up
        
        # Add refinement block (Conv 64->64 + LeakyReLU)
        # This acts as a high-resolution feature refinement before final reconstruction
        self.refine = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # Initialize weights for new refinement layer
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, skip_features=None):
        """
        Args:
            x (torch.Tensor): Encoded features (1, 64, 512, 512)
            skip_features (dict, optional): For future fusion.
        Returns:
            out (torch.Tensor): Upsampled features (1, 64, 1024, 1024)
        """
        # 1. Upsample
        out = self.upconv(x)
        out = self.pixel_shuffle(out)
        out = self.act_up(out)
        
        # 2. Refine
        out = self.refine(out)
        
        return out

def test_upsampler():
    print("Testing Multi-Scale Upsampler...")
    
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
    base_model = EnhancementModel(nf=64, num_blocks=16, scale=2).to(device)
    encoder = FeatureEncoder(base_model).to(device)
    upsampler = MultiScaleUpsampler(base_model, nf=64).to(device)
    
    img_tensor = img_tensor.to(device)

    # 4. Run Pipeline
    with torch.no_grad():
        encoded, skips = encoder(img_tensor)
        upsampled = upsampler(encoded)
    
    # 5. Verify Outputs
    print(f"Upsampled feature shape: {upsampled.shape}")
    
    if upsampled.shape == (1, 64, 1024, 1024):
        print("✓ Shape verification passed")
    else:
        print(f"X Shape verification failed. Expected (1, 64, 1024, 1024), got {upsampled.shape}")

if __name__ == "__main__":
    test_upsampler()
