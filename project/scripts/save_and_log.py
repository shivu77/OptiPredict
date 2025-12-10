import os
import sys
import cv2
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from project.scripts.preprocess import preprocess_image
from project.scripts.model_init import EnhancementModel
from project.scripts.feature_encoder import FeatureEncoder
from project.scripts.multiscale_upsample import MultiScaleUpsampler
from project.scripts.feature_fusion import MultiLevelFeatureFusion
from project.scripts.reconstruction import ReconstructionModule

def save_enhanced_image(image_array, original_filename):
    """
    Saves the enhanced image to project/outputs/enhanced/
    """
    output_dir = Path("project/outputs/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure uint8
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255.0).clip(0, 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
        
    # Construct filename
    name_stem = Path(original_filename).stem
    save_name = f"{name_stem}_SR.png"
    save_path = output_dir / save_name
    
    cv2.imwrite(str(save_path), image_array)
    return str(save_path)

def write_log(original_filename, output_path, scale=2, blocks=16):
    """
    Logs the enhancement process to project/logs/enhancement_log.txt
    """
    log_dir = Path("project/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "enhancement_log.txt"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name_stem = Path(original_filename).name
    output_name = Path(output_path).name
    
    log_entry = f"[{timestamp}] [OK] {name_stem} -> {output_name} | scale={scale}, blocks={blocks}\n"
    
    with open(log_file, "a") as f:
        f.write(log_entry)
        
    return log_file

def test_save_and_log():
    print("Testing Save and Log Module...")
    
    # 1. Setup
    input_dir = Path("project/datasets/input")
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    
    if not images:
        print("X No input images found")
        return

    test_img_path = images[0]
    print(f"Processing: {test_img_path.name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Run Pipeline (Steps 1-8 condensed)
    try:
        # Preprocess
        raw = cv2.imread(str(test_img_path))
        preprocessed = preprocess_image(raw)
        img_tensor = torch.from_numpy(np.transpose(preprocessed, (2, 0, 1))).float().unsqueeze(0).to(device)
        
        # Model
        nf = 64
        model = EnhancementModel(nf=nf, num_blocks=16, scale=2).to(device)
        encoder = FeatureEncoder(model).to(device)
        upsampler = MultiScaleUpsampler(model, nf=nf).to(device)
        fusion = MultiLevelFeatureFusion(nf=nf).to(device)
        recon = ReconstructionModule(nf=nf).to(device)
        
        with torch.no_grad():
            encoded, skips = encoder(img_tensor)
            upsampled = upsampler(encoded)
            fused = fusion(upsampled, skips)
            final_img = recon(fused) # Returns (1024, 1024, 3) uint8 or float
            
        # 3. Save
        saved_path = save_enhanced_image(final_img, test_img_path.name)
        
        # 4. Log
        log_path = write_log(test_img_path.name, saved_path)
        
        print("✓ Enhanced image saved and logged successfully")
        print(f"Output path: {saved_path}")
        print(f"Log updated at: {log_path}")
        
    except Exception as e:
        print(f"X Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_save_and_log()
