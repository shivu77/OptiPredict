import cv2
import os
import glob
import numpy as np
from pathlib import Path

def create_comparison():
    input_dir = Path("project/datasets/input")
    enhanced_dir = Path("project/outputs/enhanced")
    output_dir = Path("project/outputs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images in input directory
    input_images = list(input_dir.glob("*"))
    
    print(f"Found {len(input_images)} images in input directory.")

    for img_path in input_images:
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        filename = img_path.name
        enhanced_path = enhanced_dir / filename

        if not enhanced_path.exists():
            print(f"Enhanced image for {filename} not found.")
            continue

        # Read images
        img_in = cv2.imread(str(img_path))
        img_out = cv2.imread(str(enhanced_path))

        if img_in is None or img_out is None:
            print(f"Error reading images for {filename}")
            continue

        # Resize input to match output dimensions for side-by-side comparison
        h, w, c = img_out.shape
        img_in_resized = cv2.resize(img_in, (w, h), interpolation=cv2.INTER_CUBIC)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_in_resized, 'Original (Bicubic Upscale)', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_out, 'Real-ESRGAN Enhanced', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Concatenate side by side
        comparison = np.hstack((img_in_resized, img_out))

        # Save
        save_path = output_dir / f"compare_{filename}"
        cv2.imwrite(str(save_path), comparison)
        print(f"Created comparison: {save_path}")

if __name__ == "__main__":
    create_comparison()
