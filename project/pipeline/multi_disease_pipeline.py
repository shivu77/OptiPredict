import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import Modules
from project.scripts.preprocess import preprocess_image
from project.scripts.model_init import EnhancementModel
from project.scripts.feature_encoder import FeatureEncoder
from project.scripts.multiscale_upsample import MultiScaleUpsampler
from project.scripts.feature_fusion import MultiLevelFeatureFusion
from project.scripts.reconstruction import ReconstructionModule
from project.scripts.dr_classifier import DRClassifier
from project.scripts.four_class_classifier import classify_four_disease
from project.scripts.glaucoma_module import GlaucomaSegmentation, analyze_glaucoma

class MultiDiseasePipeline:
    def __init__(self):
        print("Initializing Multi-Disease Pipeline...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 1. Initialize Enhancement Components
        self.nf = 64
        self.base_model = EnhancementModel(nf=self.nf, num_blocks=16, scale=2).to(self.device)
        self.encoder = FeatureEncoder(self.base_model).to(self.device)
        self.upsampler = MultiScaleUpsampler(self.base_model, nf=self.nf).to(self.device)
        self.fusion = MultiLevelFeatureFusion(nf=self.nf).to(self.device)
        self.recon = ReconstructionModule(nf=self.nf).to(self.device)
        
        # 2. Initialize Diagnostic Models
        self.dr_model = DRClassifier(num_classes=5).to(self.device)
        self.dr_model.eval()
        
        self.glaucoma_segmentor = GlaucomaSegmentation() # Handles internal model/heuristic
        
        print("✓ All models initialized successfully")

    

    def run_enhancement(self, raw_image):
        """Runs the Super-Resolution Pipeline"""
        # Preprocess
        preprocessed = preprocess_image(raw_image) # (512, 512, 3)
        
        # To Tensor
        img_tensor = torch.from_numpy(np.transpose(preprocessed, (2, 0, 1))).float().unsqueeze(0).to(self.device)
        
        # Forward Pass
        with torch.no_grad():
            encoded, skips = self.encoder(img_tensor)
            upsampled = self.upsampler(encoded)
            fused = self.fusion(upsampled, skips)
            enhanced_img = self.recon(fused) # Returns numpy (1024, 1024, 3)
            
        return enhanced_img

    def analyze(self, raw_image_path):
        """
        Full end-to-end analysis:
        Raw -> Enhancement -> DR & Glaucoma Analysis
        """
        raw_image = cv2.imread(str(raw_image_path))
        if raw_image is None:
            raise ValueError(f"Could not read image: {raw_image_path}")
            
        # A. Enhancement
        start_time = time.time()
        enhanced_image = self.run_enhancement(raw_image)
        enhance_time = time.time() - start_time
        
        # B. Glaucoma Analysis
        cdr, glaucoma_status, od_mask, oc_mask = analyze_glaucoma(enhanced_image, segmentor=self.glaucoma_segmentor)
        
        label_4c, prob_4c = classify_four_disease(enhanced_image, device=self.device)
        final_disease = label_4c
        
        # E. Assembly
        result = {
            "filename": Path(raw_image_path).name,
            "enhanced_image": enhanced_image,
            "cdr_value": cdr,
            "glaucoma_status": glaucoma_status,
            "final_disease_prediction": final_disease,
            "four_class_prediction": label_4c,
            "four_class_probabilities": prob_4c,
            "performance": {
                "enhancement_time": f"{enhance_time:.2f}s"
            }
        }
        return result

def run_full_analysis(input_image_path):
    try:
        pipeline = MultiDiseasePipeline()
        print(f"\n--- Analyzing {Path(input_image_path).name} ---")
        
        results = pipeline.analyze(input_image_path)
        
        print("\n=== DIAGNOSTIC REPORT ===")
        print(f"Image: {results['filename']}")
        print(f"Enhancement Time: {results['performance']['enhancement_time']}")
        print("-" * 25)
        print(f"Glaucoma Status:      {results['glaucoma_status']}")
        print(f"Cup-to-Disc Ratio:    {results['cdr_value']}")
        print(f"4-Class Disease Prediction: {results['four_class_prediction']}")
        print(f"Confidence: {max(results['four_class_probabilities'])}")
        print(f"Final Disease Prediction: {results['final_disease_prediction']}")
        print("-" * 25)
        
        return results
        
    except Exception as e:
        print(f"X Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_pipeline():
    print("Testing Integrated Multi-Disease Pipeline...")
    
    input_dir = Path("project/datasets/input")
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not images:
        print("X No images found for testing")
        return
        
    # Pick one image
    test_img = images[0]
    
    results = run_full_analysis(test_img)
    
    if results:
        print("✓ Full Multi-Disease Pipeline Executed Successfully")

if __name__ == "__main__":
    test_pipeline()
