import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# --- U-Net Architecture Definitions ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

# --- Main Glaucoma Segmentation Class ---
class GlaucomaSegmentation:
    def __init__(self, model_path=None, use_heuristic=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(n_channels=3, n_classes=2).to(self.device)
        self.use_heuristic = use_heuristic
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.use_heuristic = False
            print(f"✓ Loaded segmentation model from {model_path}")
        else:
            # Fallback if no weights provided
            self.use_heuristic = True
            print("! No weights loaded. Using intensity-based heuristic for segmentation.")

    def predict(self, image_array):
        """
        Predicts OD and OC masks.
        Args:
            image_array: HxWx3 numpy array (BGR or RGB)
        Returns:
            od_mask, oc_mask (numpy arrays, 0-1, same size as input)
        """
        original_size = image_array.shape[:2]
        
        if self.use_heuristic:
            return self._heuristic_segmentation(image_array)
        
        # Neural Network Inference
        # 1. Resize
        img_resized = cv2.resize(image_array, (512, 512))
        
        # 2. Normalize
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 3. Forward
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # 4. Post-process
        preds = output.squeeze(0).cpu().numpy()
        od_pred = (preds[0] > 0.5).astype(np.uint8)
        oc_pred = (preds[1] > 0.5).astype(np.uint8)
        
        # 5. Resize back
        od_mask = cv2.resize(od_pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        oc_mask = cv2.resize(oc_pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        return od_mask, oc_mask

    def _heuristic_segmentation(self, image_array):
        """
        Simple heuristic for testing pipeline without trained weights.
        Assumes OD is bright and OC is brightest region within.
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (31, 31), 0)
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
        
        # Adaptive thresholds based on max intensity
        _, od_mask = cv2.threshold(blurred, maxVal * 0.65, 1, cv2.THRESH_BINARY)
        _, oc_mask = cv2.threshold(blurred, maxVal * 0.90, 1, cv2.THRESH_BINARY)
        
        # Cleanup
        kernel = np.ones((5,5), np.uint8)
        od_mask = cv2.morphologyEx(od_mask, cv2.MORPH_OPEN, kernel)
        oc_mask = cv2.morphologyEx(oc_mask, cv2.MORPH_OPEN, kernel)
        
        return od_mask.astype(np.uint8), oc_mask.astype(np.uint8)

def compute_cdr(od_mask, oc_mask):
    """
    Computes Vertical Cup-to-Disc Ratio.
    """
    def get_vertical_height(mask):
        coords = cv2.findNonZero(mask)
        if coords is None: return 0
        x, y, w, h = cv2.boundingRect(coords)
        return h

    h_disc = get_vertical_height(od_mask)
    h_cup = get_vertical_height(oc_mask)
    
    if h_disc == 0: return 0.0
    
    cdr = h_cup / h_disc
    return round(cdr, 3)

def analyze_glaucoma(enhanced_image_array, segmentor=None):
    """
    Full Glaucoma Analysis Pipeline.
    """
    if segmentor is None:
        segmentor = GlaucomaSegmentation()
        
    od_mask, oc_mask = segmentor.predict(enhanced_image_array)
    cdr = compute_cdr(od_mask, oc_mask)
    
    # Decision Logic
    status = "Glaucoma Suspected" if cdr > 0.6 else "Normal"
    
    return cdr, status, od_mask, oc_mask

def test_glaucoma_module():
    print("Testing Glaucoma Segmentation Module...")
    
    input_dir = Path("project/outputs/enhanced")
    images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    
    if not images:
        print("X No enhanced images found")
        return
        
    img_path = images[0]
    print(f"Processing: {img_path.name}")
    img = cv2.imread(str(img_path))
    
    if img is None:
        print("X Failed to read image")
        return
        
    cdr, status, od_mask, oc_mask = analyze_glaucoma(img)
    
    print("-" * 30)
    print(f"CDR: {cdr}")
    print(f"Glaucoma Status: {status}")
    print(f"OD Mask shape: {od_mask.shape}")
    print(f"OC Mask shape: {oc_mask.shape}")
    print("-" * 30)
    
    if od_mask.max() > 0:
        print("✓ Masks generated successfully")
    else:
        print("! Masks empty (Heuristic might need tuning for this image)")
        
    print("✓ Glaucoma module test complete")

if __name__ == "__main__":
    test_glaucoma_module()
