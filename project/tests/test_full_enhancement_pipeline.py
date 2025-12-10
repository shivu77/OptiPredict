import torch
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from project.scripts.preprocess import preprocess_image
from project.scripts.model_init import EnhancementModel
from project.scripts.feature_encoder import FeatureEncoder
from project.scripts.multiscale_upsample import MultiScaleUpsampler
from project.scripts.feature_fusion import MultiLevelFeatureFusion
from project.scripts.reconstruction import ReconstructionModule

def test_full_pipeline():
    print("Starting End-to-End Enhancement Pipeline Test...")
    
    # 1. Setup Paths and Device
    input_dir = Path("project/datasets/input")
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    
    if not images:
        print("X No input images found in project/datasets/input")
        return

    test_img_path = images[0]
    print(f"Processing image: {test_img_path.name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Preprocessing
    try:
        raw_img = cv2.imread(str(test_img_path))
        if raw_img is None:
            print(f"X Failed to read image: {test_img_path}")
            return
            
        preprocessed_img = preprocess_image(raw_img) # Returns (512, 512, 3) float32 [0,1]
        print(f"✓ Preprocessing complete. Shape: {preprocessed_img.shape}")
        
        # Convert to Tensor (B, C, H, W)
        img_tensor = torch.from_numpy(np.transpose(preprocessed_img, (2, 0, 1))).float().unsqueeze(0).to(device)
    except Exception as e:
        print(f"X Preprocessing failed: {e}")
        return

    # 3. Initialize Model Components
    try:
        nf = 64
        base_model = EnhancementModel(nf=nf, num_blocks=16, scale=2).to(device)
        
        encoder = FeatureEncoder(base_model).to(device)
        upsampler = MultiScaleUpsampler(base_model, nf=nf).to(device)
        fusion = MultiLevelFeatureFusion(nf=nf).to(device)
        reconstruction = ReconstructionModule(nf=nf).to(device)
        
        print("✓ Model components initialized")
    except Exception as e:
        print(f"X Model initialization failed: {e}")
        return

    # 4. Execute Pipeline
    try:
        with torch.no_grad():
            # A. Encode
            encoded_features, skip_features = encoder(img_tensor)
            
            # B. Upsample
            upsampled_features = upsampler(encoded_features)
            
            # C. Fuse
            fused_features = fusion(upsampled_features, skip_features)
            
            # D. Reconstruct
            final_image = reconstruction(fused_features)
            
        print("✓ Pipeline execution complete")
        print(f"Final enhanced image shape: {final_image.shape}")
        
        if final_image.shape == (1024, 1024, 3):
            print("✓ End-to-end enhancement pipeline executed successfully")
        else:
            print(f"X Output shape mismatch. Expected (1024, 1024, 3), got {final_image.shape}")
            
    except Exception as e:
        print(f"X Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_pipeline()
