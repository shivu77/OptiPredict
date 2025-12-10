import os
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import sys

def check_class_balance():
    print("=== Checking Dataset Class Balance ===")
    
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("project/datasets")
    
    # Initialize counts
    counts = {
        "Normal": 0,
        "Cataract": 0,
        "Diabetic Retinopathy": 0,
        "Glaucoma": 0
    }
    
    # Track files to avoid double counting (if same file referenced in json and folder scan)
    processed_files = set()
    processed_names = set()
    
    # 1. Scan for labels.json files
    print("\nScanning for labels.json files...")
    json_files = list(dataset_root.rglob("labels.json"))
    
    for json_file in json_files:
        print(f"  Found: {json_file}")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for filename, attributes in data.items():
                # Construct a unique ID for the file to avoid duplicates
                # Assuming filename is unique within the json context, but let's try to resolve full path if possible
                # If not, just use filename as key
                file_id = f"{json_file.parent}/{filename}"
                if file_id in processed_files:
                    continue
                processed_files.add(file_id)
                processed_names.add(Path(filename).name)
                
                # Determine class based on priority logic (Cataract > DR > Glaucoma > Normal)
                # Check for keys in attributes
                if isinstance(attributes, dict):
                    dr_label = attributes.get("dr_label", "No_DR")
                    glaucoma_status = attributes.get("glaucoma_status", "Normal")
                    cdr_value = attributes.get("cdr_value")
                    cataract_label = attributes.get("cataract_label", "Normal")
                    final_label = "Normal"
                    if cataract_label == "Cataract":
                        final_label = "Cataract"
                    elif dr_label == "DR":
                        final_label = "Diabetic Retinopathy"
                    elif glaucoma_status == "Suspected" or (cdr_value is not None and cdr_value >= 0.6):
                        final_label = "Glaucoma"
                else:
                    val = str(attributes)
                    if val.lower() == "dr":
                        final_label = "Diabetic Retinopathy"
                    elif val.lower() == "normal":
                        final_label = "Normal"
                    elif val.lower() == "cataract":
                        final_label = "Cataract"
                    elif val.lower() == "glaucoma":
                        final_label = "Glaucoma"
                    else:
                        final_label = "Normal"
                
                counts[final_label] += 1
                
        except Exception as e:
            print(f"  ! Error reading {json_file}: {e}")

    # 2. Scan for Folder-based labels (e.g. datasets/train/Cataract/img.jpg)
    print("\nScanning for folder-based labels...")
    # Define mapping from folder names to standard class names
    folder_map = {
        "normal": "Normal",
        "cataract": "Cataract",
        "dr": "Diabetic Retinopathy",
        "diabetic_retinopathy": "Diabetic Retinopathy",
        "glaucoma": "Glaucoma"
    }
    
    for path in dataset_root.rglob("*"):
        if path.is_dir() and path.name.lower() in folder_map:
            label = folder_map[path.name.lower()]
            # Count images in this folder
            images = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
            
            for img in images:
                if str(img) in processed_files or img.name in processed_names:
                    continue
                processed_files.add(str(img))
                counts[label] += 1
                
            if len(images) > 0:
                print(f"  Found folder '{path.name}' with {len(images)} images -> Mapped to {label}")

    # 3. Summary and Analysis
    total_images = sum(counts.values())
    print("\n" + "="*30)
    print("CLASS DISTRIBUTION REPORT")
    print("="*30)
    
    if total_images == 0:
        print("No labeled images found in project/datasets/.")
        return

    max_count = 0
    min_count = total_images
    
    for label, count in counts.items():
        percentage = (count / total_images) * 100
        print(f"{label:<20}: {count} images ({percentage:.1f}%)")
        
        if count > max_count:
            max_count = count
        if count < min_count and count > 0: # Ignore 0 counts for ratio calculation to avoid div by zero? 
                                            # Prompt says "Max class count / Min class count". 
                                            # If a class has 0, ratio is infinite.
            min_count = count
            
    # Handle min_count = 0 for ratio
    missing_classes = [label for label, count in counts.items() if count == 0]
    
    if min_count == 0 or min_count == total_images:
        # If min is 0 (should be caught by the count > 0 check unless all are 0)
        # Or if only one class is present
        imbalance_ratio = 999.0 
        real_min = min_count
    else:
        imbalance_ratio = max_count / min_count
        real_min = min_count

    print("-" * 30)
    print(f"Total Images:    {total_images}")
    print(f"Max Class Count: {max_count}")
    print(f"Min Class Count: {real_min} (excluding zeros)")
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
    
    if missing_classes:
        print(f"\n⚠ CRITICAL: The following classes have 0 images: {', '.join(missing_classes)}")
        print("  The model will effectively never learn these classes.")
    
    if imbalance_ratio > 3 or missing_classes:
        print("\n⚠ Dataset highly imbalanced or incomplete.")
    else:
        print("\n✓ Dataset balanced enough for multi-class training.")
        
    # 4. Visualization
    try:
        output_plot = Path("project/analysis/class_distribution.png")
        plt.figure(figsize=(10, 6))
        bars = plt.bar(counts.keys(), counts.values(), color=['green', 'gray', 'red', 'blue'])
        
        # Add counts on top
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')
            
        plt.title(f"Disease Class Distribution (N={total_images})")
        plt.xlabel("Disease Class")
        plt.ylabel("Number of Images")
        plt.ylim(0, max_count * 1.2) # Add some headroom
        
        plt.savefig(output_plot)
        print(f"\n✓ Distribution plot saved to: {output_plot}")
        
    except Exception as e:
        print(f"\n! Could not generate plot: {e}")

if __name__ == "__main__":
    check_class_balance()
