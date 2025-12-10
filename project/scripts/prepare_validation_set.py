import os
import shutil
import json
from pathlib import Path
import random

def prepare_validation_dataset():
    print("Preparing Validation Dataset...")
    
    # 1. Setup Directories
    base_path = Path(__file__).resolve().parents[2]
    input_dir = base_path / "project/datasets/input"
    val_dir = base_path / "project/datasets/validation"
    val_img_dir = val_dir / "images"
    
    # Create directories
    val_img_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {val_img_dir}")
    
    # 2. Select Images
    all_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    
    if not all_images:
        print("X No input images found to copy.")
        return
        
    # Use all available images (since we have few), or limit if many
    selected_images = all_images[:20] 
    
    # 3. Copy Images & Create Labels
    labels = {}
    
    print(f"Copying {len(selected_images)} images...")
    
    for img_path in selected_images:
        # Copy file
        dest_path = val_img_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        
        # Generate synthetic label (since we don't have real ground truth metadata handy)
        # In a real scenario, this would come from a CSV or dataset annotations
        is_dr = random.choice([True, False])
        is_glaucoma = random.choice([True, False])
        
        labels[img_path.name] = {
            "dr_label": "DR" if is_dr else "No_DR",
            "glaucoma_status": "Suspected" if is_glaucoma else "Normal",
            "cdr_value": round(random.uniform(0.6, 0.9), 2) if is_glaucoma else round(random.uniform(0.3, 0.55), 2)
        }
        
    # 4. Write labels.json
    json_path = val_dir / "labels.json"
    with open(json_path, 'w') as f:
        json.dump(labels, f, indent=4)
        
    print(f"✓ Created labels.json with {len(labels)} entries")
    
    # 5. Verification Script
    print("\nRunning Verification...")
    
    if not json_path.exists():
        print("X labels.json missing")
        return
        
    with open(json_path, 'r') as f:
        loaded_labels = json.load(f)
        
    missing_files = []
    for filename in loaded_labels.keys():
        file_path = val_img_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
            
    if missing_files:
        print(f"X Missing files in validation folder: {missing_files}")
    else:
        print("✓ Validation dataset and labels prepared successfully")
        print(f"  Location: {val_dir}")

if __name__ == "__main__":
    prepare_validation_dataset()
