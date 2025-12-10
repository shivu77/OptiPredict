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
from project.scripts.multiscale_upsample import MultiScaleUpsampler
from project.scripts.feature_fusion import MultiLevelFeatureFusion

class ReconstructionModule(nn.Module):
    def __init__(self, nf=64, out_c=3):
        """
        Reconstructs final RGB image from fused features.
        Applies post-processing refinement.
        """
        super(ReconstructionModule, self).__init__()
        
        # Final convolution: 64 -> 3
        self.conv_last = nn.Conv2d(nf, out_c, 3, 1, 1, bias=True)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv_last.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if self.conv_last.bias is not None:
            nn.init.constant_(self.conv_last.bias, 0)

    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, 64, 1024, 1024)
        Returns:
            enhanced_img_array: (1024, 1024, 3) numpy array (uint8)
        """
        # 1. Final Convolution
        out = self.conv_last(fused_features)
        
        # 2. Clamp to [0, 1]
        out = torch.clamp(out, 0, 1)
        
        # 3. Convert to Numpy (H, W, C)
        # Assuming batch size 1 for reconstruction step as per requirement
        img_tensor = out.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        
        # 4. Post-processing: Color Correction & Texture Refinement
        # Convert to uint8 for OpenCV operations
        img_uint8 = (img_tensor * 255.0).clip(0, 255).astype(np.uint8)
        
        # A. Light Color Correction (LAB CLAHE) - Optional but beneficial for retina
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # B. Light Texture Refinement (Unsharp Masking)
        gaussian = cv2.GaussianBlur(img_corrected, (0, 0), 2.0)
        img_refined = cv2.addWeighted(img_corrected, 1.1, gaussian, -0.1, 0)
        
        return img_refined

def test_reconstruction():
    print("Testing Reconstruction Module...")
    
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
    reconstruction = ReconstructionModule(nf=nf).to(device)
    
    img_tensor = img_tensor.to(device)

    # 4. Run Pipeline
    with torch.no_grad():
        encoded, skips = encoder(img_tensor)
        upsampled = upsampler(encoded)
        fused = fusion(upsampled, skips)
        final_img = reconstruction(fused)
    
    # 5. Verify Outputs
    print(f"Reconstructed image shape: {final_img.shape}")
    
    if final_img.shape == (1024, 1024, 3):
        print("✓ Shape verification passed")
    else:
        print(f"X Shape verification failed. Expected (1024, 1024, 3), got {final_img.shape}")

if __name__ == "__main__":
    test_reconstruction()
