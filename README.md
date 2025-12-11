# OptiPredict — AI-Powered Retinal Disease Detection System

## Quick Start

- Clone: `git clone https://github.com/shivu77/OptiPredict.git && cd OptiPredict`
- Create env (Windows PowerShell): `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
- Install core deps:
  - CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  - GPU (CUDA 11.8): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
  - Common: `pip install opencv-python numpy basicsr`
  - Optional Real-ESRGAN extras: `pip install -r project/real_esrgan/requirements.txt`
- Prepare dataset layout under `project/datasets/balanced_dataset/`:
  - `Normal/`, `DR/`, `Cataract/`, `Glaucoma/` with images inside each class folder
- Train the 4-class model:
  - ResNet18: `python project/training/train_4_class_classifier.py --epochs 15 --batch_size 32 --arch resnet18 --workers 0 --data_root "project/datasets/balanced_dataset"`
  - EfficientNet-B0: `python project/training/train_4_class_classifier.py --epochs 15 --batch_size 32 --arch efficientnet_b0 --workers 0 --data_root "project/datasets/balanced_dataset"`
- Run detector: `python project/run_disease_detector.py project/datasets/validation/images/0_left.jpg`

## What You’ll See

- Pipeline confirms device and model init; prints `Loaded 4-class model successfully.` when `project/models/four_class_classifier.pth` exists.
- Detector outputs final disease label, confidence, and 4-class probabilities.
- Visuals saved to `project/outputs/visuals/`:
  - DR Grad-CAM: `<filename>_dr_heatmap.png`
  - Glaucoma OD/OC overlay: `<filename>_glaucoma_overlay.png`

## Useful Commands

- Integrity test: `python project/tests/test_project_integrity.py`
- Pipeline test: `python project/tests/test_full_multi_disease_pipeline.py`
- Accuracy vs labels: `python project/tests/test_prediction_accuracy.py`

## Troubleshooting

- If predictions seem wrong, ensure `project/models/four_class_classifier.pth` exists and matches the selected `--arch`.
- On Windows, use `--workers 0` to avoid DataLoader issues.
- The glaucoma message `No weights loaded` refers to segmentation heuristics; it does not affect the classifier.
