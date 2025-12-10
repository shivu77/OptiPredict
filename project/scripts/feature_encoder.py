import torch
import torch.nn as nn
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from project.scripts.model_init import EnhancementModel

class FeatureEncoder(nn.Module):
    def __init__(self, model):
        """
        Wraps the EnhancementModel to act as a feature encoder.
        Extracts low, mid, and high-level features.
        """
        super(FeatureEncoder, self).__init__()
        self.model = model

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
        Returns:
            encoded_features (torch.Tensor): Final encoded features before upsampling.
            skip_features (dict): Dictionary containing intermediate features.
        """
        # 1. Initial Convolution (Low-level features)
        feat_first = self.model.conv_first(x)
        
        # 2. Residual Blocks (Mid & High-level features)
        # We iterate through the body to extract intermediate features
        body_feat = feat_first
        mid_feat = None
        
        # Assuming self.model.body is a nn.Sequential of ResidualBlocks
        total_blocks = len(self.model.body)
        mid_point = total_blocks // 2
        
        for i, block in enumerate(self.model.body):
            body_feat = block(body_feat)
            if i == mid_point - 1:
                mid_feat = body_feat  # Capture mid-level features
        
        # 3. Post-Residual Convolution
        feat_body_out = self.model.conv_body(body_feat)
        
        # 4. Global Skip Connection
        encoded_features = feat_first + feat_body_out
        
        skip_features = {
            'low_level': feat_first,
            'mid_level': mid_feat
        }
        
        return encoded_features, skip_features

def test_encoder():
    print("Testing Feature Encoder...")
    
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
    # HWC to CHW -> (1, 3, 512, 512)
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
    
    # 3. Initialize Model and Encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = EnhancementModel(nf=64, num_blocks=16, scale=2).to(device)
    encoder = FeatureEncoder(base_model).to(device)
    
    img_tensor = img_tensor.to(device)

    # 4. Run Forward Pass
    with torch.no_grad():
        encoded, skips = encoder(img_tensor)
    
    # 5. Verify Outputs
    print(f"Encoded feature shape: {encoded.shape}")
    print(f"Skip connections created: {len(skips)} ({list(skips.keys())})")
    
    if encoded.shape == (1, 64, 512, 512):
        print("✓ Shape verification passed")
    else:
        print(f"X Shape verification failed. Expected (1, 64, 512, 512), got {encoded.shape}")

if __name__ == "__main__":
    test_encoder()
