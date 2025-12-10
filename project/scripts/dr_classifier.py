import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        """
        Initializes the DR Classifier using a ResNet18 backbone.
        Args:
            num_classes (int): Number of output classes (default 5 for DR grades).
                               0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
            pretrained (bool): Whether to use ImageNet pretrained weights.
        """
        super(DRClassifier, self).__init__()
        
        # Use ResNet18 as the backbone (lightweight and effective)
        # Handling different torchvision versions for weights
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
        except ImportError:
            self.model = models.resnet18(pretrained=pretrained)
            
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # Define normalization (ImageNet standards)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        return self.model(x)
    
    def load_weights(self, weight_path):
        """Placeholder for loading trained weights"""
        if Path(weight_path).exists():
            self.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print(f"✓ Loaded weights from {weight_path}")
        else:
            print(f"! Warning: Weight file {weight_path} not found. Using random/pretrained initialization.")

def classify_dr(enhanced_image_array, model=None, device=None):
    """
    Classifies an enhanced retinal image for Diabetic Retinopathy.
    
    Args:
        enhanced_image_array (numpy.ndarray): Input image (H, W, 3), uint8 or float.
        model (DRClassifier, optional): Loaded model instance.
        device (torch.device, optional): Device to run inference on.
        
    Returns:
        predicted_class (int): Predicted DR grade (0-4).
        probabilities (list): Probability for each class.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if model is None:
        model = DRClassifier(num_classes=5).to(device)
        model.eval()
    
    # 1. Preprocess: Resize to 224x224 (standard ResNet input)
    input_size = (224, 224)
    
    # Ensure image is uint8 for consistency before resizing/normalization logic
    if enhanced_image_array.dtype != np.uint8:
        if enhanced_image_array.max() <= 1.0:
            enhanced_image_array = (enhanced_image_array * 255).astype(np.uint8)
        else:
            enhanced_image_array = enhanced_image_array.astype(np.uint8)
            
    resized_img = cv2.resize(enhanced_image_array, input_size, interpolation=cv2.INTER_AREA)
    
    # 2. Convert to Tensor and Normalize
    # Convert BGR to RGB (if using cv2.imread upstream, usually BGR. 
    # But our pipeline output might be RGB. Let's assume RGB for 'enhanced_image_array' 
    # as per previous reconstruction module which converts LAB->BGR then likely kept as BGR or converted.
    # Wait, reconstruction module outputs: cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).
    # So input is BGR.
    # ResNet expects RGB.
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # To Tensor [0, 1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # Normalize
    img_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(img_tensor)
    
    img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension
    
    # 3. Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    return predicted_class, probabilities[0].cpu().numpy().tolist()

def generate_gradcam(model, image_array, target_class=None, device=None):
    """
    Generates a Grad-CAM heatmap for the given image and model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    model.eval()
    
    # Preprocess (Same as classify_dr)
    input_size = (224, 224)
    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
            
    resized_img = cv2.resize(image_array, input_size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(img_tensor).unsqueeze(0).to(device)
    
    # Hooks
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    def forward_hook(module, input, output):
        activations.append(output)
        
    # Register hooks on the last conv layer
    # For ResNet18, it's layer4
    target_layer = model.model.layer4
    
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward
    model.zero_grad()
    output = model(img_tensor)
    
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()
        
    # Backward
    score = output[:, target_class]
    score.backward()
    
    # Generate CAM
    grads = gradients[0].cpu().data.numpy()[0] # (512, 7, 7)
    fmaps = activations[0].cpu().data.numpy()[0] # (512, 7, 7)
    
    weights = np.mean(grads, axis=(1, 2)) # (512,)
    cam = np.zeros(fmaps.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * fmaps[i]
        
    cam = np.maximum(cam, 0)
    # Resize to original image size
    cam = cv2.resize(cam, (image_array.shape[1], image_array.shape[0])) 
    cam = cam - np.min(cam)
    cam_max = np.max(cam)
    if cam_max > 0:
        cam = cam / cam_max
        
    # Clean up hooks
    handle_f.remove()
    handle_b.remove()
    
    # Convert to heatmap (uint8 0-255)
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

def test_dr_classifier():
    print("Testing DR Classification Module...")
    
    # 1. Locate an enhanced image
    enhanced_dir = Path("project/outputs/enhanced")
    images = list(enhanced_dir.glob("*.png")) + list(enhanced_dir.glob("*.jpg"))
    
    if not images:
        print("X No enhanced images found in project/outputs/enhanced/")
        return

    test_img_path = images[0]
    print(f"Loading image: {test_img_path}")
    
    # 2. Load Image
    img = cv2.imread(str(test_img_path)) # Loads as BGR
    if img is None:
        print(f"X Failed to read {test_img_path}")
        return
        
    # 3. Run Classification
    print("Initializing classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = DRClassifier(num_classes=5).to(device)
    classifier.eval()
    
    print(f"Running inference on device: {device}")
    pred_class, probs = classify_dr(img, model=classifier, device=device)
    
    # 4. Results
    dr_grades = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
    print("-" * 30)
    print(f"Prediction: Class {pred_class} ({dr_grades.get(pred_class, 'Unknown')})")
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  Grade {i}: {p:.4f}")
    print("-" * 30)
    print("✓ DR Classification test complete")

if __name__ == "__main__":
    test_dr_classifier()
