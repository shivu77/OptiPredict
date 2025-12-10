import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add project root to path so we can import scripts
sys.path.append(str(Path(__file__).resolve().parents[2]))

from project.scripts.preprocess import preprocess_image
from project.scripts.model_init import EnhancementModel

def run_verification():
    print("Starting verification...")

    # --- STEP 1: Environment & Weights ---
    weights_path = Path("project/real_esrgan/weights/RealESRGAN_x2plus.pth")
    if weights_path.exists():
        print("✓ Weights found")
    else:
        print(f"X Weights NOT found at {weights_path}")
        return

    # --- STEP 2: Preprocessing Module ---
    input_dir = Path("project/datasets/input")
    # Find first image
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_path = None
    for ext in extensions:
        found = list(input_dir.glob(ext))
        if found:
            image_path = found[0]
            break
    
    if image_path:
        img = cv2.imread(str(image_path))
        if img is not None:
            processed = preprocess_image(img)
            
            if processed.shape == (512, 512, 3):
                # Save temporary output
                test_out_dir = Path("project/datasets/preprocessed_test")
                test_out_dir.mkdir(parents=True, exist_ok=True)
                save_path = test_out_dir / f"test_{image_path.name}"
                
                # Convert back to uint8 for saving
                img_save = (processed * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(save_path), img_save)
                
                print("✓ Preprocessing successful")
            else:
                print(f"X Preprocessing failed: Shape is {processed.shape}, expected (512, 512, 3)")
                return
        else:
             print(f"X Could not read image {image_path}")
             return
    else:
        print("X No images found in project/datasets/input/ to test preprocessing")
        return

    # --- STEP 3: Model Initialization Module ---
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnhancementModel(nf=64, num_blocks=16, scale=2).to(device)
        
        # Basic check: count parameters or check string
        # num_params = sum(p.numel() for p in model.parameters())
        # print(f"Model parameters: {num_params}")
        
        print("✓ Model initialization successful")
    except Exception as e:
        print(f"X Model initialization failed: {e}")
        return

    print("ALL PREVIOUS STEPS VERIFIED — READY FOR STEP 4")

if __name__ == "__main__":
    run_verification()
