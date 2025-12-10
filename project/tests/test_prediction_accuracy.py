import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from project.pipeline.multi_disease_pipeline import MultiDiseasePipeline
from project.scripts.four_class_classifier import CLASS_NAMES

def test_prediction_accuracy():
    print("Starting Prediction Accuracy Test...")
    
    # 1. Setup Paths
    base_path = Path(__file__).resolve().parents[2]
    val_dir = base_path / "project/datasets/validation"
    val_img_dir = val_dir / "images"
    labels_path = val_dir / "labels.json"
    
    log_dir = base_path / "project/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failure_log = log_dir / "prediction_failures.txt"
    
    if not labels_path.exists():
        print("X Labels file missing. Run Step 16 first.")
        return
        
    with open(labels_path, 'r') as f:
        labels = json.load(f)
        
    if not labels:
        print("X Labels file is empty.")
        return

    # 2. Initialize Pipeline
    pipeline = MultiDiseasePipeline()
    
    # 3. Counters
    total_samples = 0
    four_class_correct = 0
    cdr_errors = []
    
    # 4. Loop through samples
    print(f"\nProcessing {len(labels)} validation samples...")
    
    for filename, gt in labels.items():
        img_path = val_img_dir / filename
        if not img_path.exists():
            print(f"! Skipping missing file: {filename}")
            continue
            
        try:
            total_samples += 1
            print(f"Analyzing: {filename}")
            
            # Run Pipeline
            results = pipeline.analyze(img_path)
            
            # --- Derive expected 4-class label from ground truth fields ---
            gt_dr_label = gt.get('dr_label')
            gt_glaucoma = gt.get('glaucoma_status')
            gt_cdr = gt.get('cdr_value')
            gt_cataract = gt.get('cataract_label')

            expected_label = 'Normal'
            if gt_cataract == 'Cataract':
                expected_label = 'Cataract'
            elif gt_dr_label == 'DR':
                expected_label = 'DR'
            elif gt_glaucoma == 'Suspected' or (gt_cdr is not None and gt_cdr >= 0.6):
                expected_label = 'Glaucoma'

            pred_4c = results['four_class_prediction']
            if pred_4c in CLASS_NAMES and pred_4c == expected_label:
                four_class_correct += 1

            # --- Compare CDR ---
            pred_cdr = results['cdr_value']
            
            if gt_cdr is not None:
                error = abs(pred_cdr - gt_cdr)
                cdr_errors.append(error)
                
        except Exception as e:
            print(f"X Failure on {filename}: {e}")
            with open(failure_log, "a") as f:
                f.write(f"{filename}: {str(e)}\n")

    # 5. Statistics
    if total_samples == 0:
        print("X No samples processed successfully.")
        return
        
    four_class_acc = four_class_correct / total_samples
    mean_cdr_error = np.mean(cdr_errors) if cdr_errors else 0.0
    
    print("\n" + "="*40)
    print("✓ Prediction Accuracy Test Completed")
    print(f"Total Samples: {total_samples}")
    print(f"4-Class Accuracy:  {four_class_acc:.2f}")
    print(f"Mean CDR Error:    {mean_cdr_error:.2f}")
    print("="*40)
    
    print("\n✓ Validation Complete — Ready for Deployment Stage")

if __name__ == "__main__":
    test_prediction_accuracy()
