import os
import sys
from pathlib import Path
import urllib.request
import ssl
import cv2
import torch
from torchvision.transforms import functional as F
import types

# Monkey patch for basicsr compatibility with newer torchvision
try:
    from torchvision.transforms import functional_tensor
except ImportError:
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR / "real_esrgan"
sys.path.insert(0, str(REPO_DIR))

from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

MODEL_NAME = "RealESRGAN_x2plus.pth"
MODEL_PRIMARY_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
MODEL_FALLBACK_URL = "https://huggingface.co/dtarnow/UPscaler/resolve/main/RealESRGAN_x2plus.pth"

def ensure_model(model_path: Path):
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_PRIMARY_URL
    try:
        ssl_ctx = ssl.create_default_context()
        urllib.request.urlretrieve(url, str(model_path))
    except Exception:
        url = MODEL_FALLBACK_URL
        ssl_ctx = ssl.create_default_context()
        urllib.request.urlretrieve(url, str(model_path))

def list_images(input_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    files = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix in exts:
            files.append(p)
    return files

def main():
    input_dir = PROJECT_DIR / "datasets" / "input"
    output_dir = PROJECT_DIR / "outputs" / "enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = REPO_DIR / "weights"
    model_path = weights_dir / MODEL_NAME
    ensure_model(model_path)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    half_precision = bool(torch.cuda.is_available())
    upsampler = RealESRGANer(scale=2, model_path=str(model_path), model=model, tile=0, pre_pad=0, half=half_precision)

    images = list_images(input_dir)
    for img_path in images:
        print(f"Processing {img_path.name}")
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        output, _ = upsampler.enhance(img, outscale=2)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), output)
        print(f"Enhanced {img_path.name} saved.")

if __name__ == "__main__":
    main()

