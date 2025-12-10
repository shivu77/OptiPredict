import sys
import cv2
import numpy as np
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from project.pipeline.multi_disease_pipeline import MultiDiseasePipeline
from project.scripts.four_class_classifier import classify_four_disease
from project.scripts.dr_classifier import generate_gradcam
from project.scripts.glaucoma_module import analyze_glaucoma

def main():
    print("=== Interactive 4-Disease Detector ===")
    
    # 1. Ask User Input
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        try:
            input_path = input("Enter retinal image path: ").strip()
        except EOFError:
             print("No input provided via stdin. Usage: python run_disease_detector.py <image_path>")
             return

    # Remove quotes if user added them
    input_path = input_path.strip('"').strip("'")
    
    if not Path(input_path).exists():
        print(f"X Error: File not found at {input_path}")
        return

    # 2. Enhancement Pipeline
    print("\nRunning Enhancement Pipeline...")
    pipeline = MultiDiseasePipeline()
    # run_full_analysis is a standalone function, but we want to use our instance.
    # So we use pipeline.analyze()
    results = pipeline.analyze(input_path)
    
    if results is None:
        print("Pipeline failed.")
        return
        
    enhanced_image = results["enhanced_image"]
    
    label_4c, prob_4c = classify_four_disease(enhanced_image, device=pipeline.device)
    final_label = label_4c
    final_conf = max(prob_4c)

    # 5. Visual Explanations
    visuals_dir = Path("project/outputs/visuals")
    visuals_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(input_path).stem
    
    print("\nGenerating Visual Explanations...")
    
    if final_label == "DR":
        try:
            heatmap = generate_gradcam(pipeline.dr_model, enhanced_image, device=pipeline.device)
            # Overlay
            heatmap_resized = cv2.resize(heatmap, (enhanced_image.shape[1], enhanced_image.shape[0]))
            overlay = cv2.addWeighted(enhanced_image, 0.7, heatmap_resized, 0.3, 0)
            
            save_path = visuals_dir / f"{filename}_dr_heatmap.png"
            cv2.imwrite(str(save_path), overlay)
            print(f"✓ Saved DR Grad-CAM: {save_path}")
        except Exception as e:
            print(f"! Failed to generate Grad-CAM: {e}")
            
    if final_label == "Glaucoma":
        try:
            cdr_value, glaucoma_status, od_mask, oc_mask = analyze_glaucoma(enhanced_image, segmentor=pipeline.glaucoma_segmentor)
            overlay = enhanced_image.copy()
            
            # Draw OD contours (Green)
            if od_mask is not None:
                contours_od, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours_od, -1, (0, 255, 0), 2)
            
            # Draw OC contours (Blue)
            if oc_mask is not None:
                contours_oc, _ = cv2.findContours(oc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours_oc, -1, (255, 0, 0), 2)
            
            save_path = visuals_dir / f"{filename}_glaucoma_overlay.png"
            cv2.imwrite(str(save_path), overlay)
            print(f"✓ Saved Glaucoma Overlay: {save_path}")
        except Exception as e:
            print(f"! Failed to generate Glaucoma Overlay: {e}")

    # 6. Final Output to Terminal
    print("\n" + "="*30)
    print(f"FINAL DIAGNOSIS: {final_label}")
    print(f"Confidence:      {final_conf:.4f}")
    print("-" * 30)
    print(f"4-Class Probabilities: {prob_4c}")
    print("="*30)
    print(f"Visual explanations saved in {visuals_dir}/")

if __name__ == "__main__":
    main()
    print("✓ Unified 4-Disease Interactive Detector Operational")
