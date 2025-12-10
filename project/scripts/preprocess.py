import cv2
import numpy as np
import os
from pathlib import Path

def preprocess_images():
    input_dir = Path("project/datasets/input")
    output_dir = Path("project/datasets/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extensions to look for
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(ext))
        # Also check for uppercase extensions just in case
        files.extend(input_dir.glob(ext.upper()))
    
    # Remove duplicates if any (e.g. case sensitivity issues)
    files = sorted(list(set(files)))

    print(f"Found {len(files)} images in {input_dir}")

    for file_path in files:
        try:
            # 1. Read RGB image
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"Warning: Could not read {file_path}")
                continue

            # Process image
            img_normalized = preprocess_image(img)
            
            # For saving to disk as an image, we convert back to uint8 [0, 255]
            img_save = (img_normalized * 255).clip(0, 255).astype(np.uint8)

            # 5. Save processed image
            output_path = output_dir / file_path.name
            cv2.imwrite(str(output_path), img_save)
            
            print(f"Processed: {file_path.name} -> {output_path}")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

def preprocess_image(img):
    # CLAHE setup: Clip limit 2.0, Grid size 8x8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 2. Apply CLAHE on L-channel (LAB space)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # 3. Resize to 512x512 using cubic interpolation
    target_size = (512, 512)
    img_resized = cv2.resize(img_clahe, target_size, interpolation=cv2.INTER_CUBIC)

    # 4. Normalize pixel values to [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized

if __name__ == "__main__":
    preprocess_images()
