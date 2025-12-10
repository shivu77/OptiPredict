import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models
from pathlib import Path

CLASS_NAMES = ["Normal", "DR", "Cataract", "Glaucoma"]
WEIGHTS_PATH = Path("project/models/four_class_classifier.pth")

def build_model(arch: str):
    if arch == "resnet18":
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 4)
        return model
    else:
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 4)
        return model

def detect_arch_from_state(state):
    keys = list(state.keys())
    if any(k.startswith("layer1.") or k.startswith("fc.") for k in keys):
        return "resnet18"
    return "efficientnet_b0"

def classify_four_disease(enhanced_image_array, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch = "resnet18"
    state = None
    if WEIGHTS_PATH.exists():
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        arch = detect_arch_from_state(state)
    model = build_model(arch).to(device)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    img = enhanced_image_array
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy().tolist()
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
    return label, probs
