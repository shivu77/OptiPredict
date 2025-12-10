# OptiPredict — AI-Powered Retinal Disease Detection System

## 🧠 Overview
OptiPredict is an AI-based retinal imaging system that detects four major eye diseases using fundus photographs:

- Normal
- Diabetic Retinopathy (DR)
- Cataract
- Glaucoma

The system enhances images using a custom ESRGAN-based pipeline, then applies a unified 4-class deep neural network classifier and disease-specific visual explanations.

## 🚀 Features

### ✔ Image Enhancement (Real-ESRGAN)
- Multi-scale feature extraction
- Upsampling + fusion
- Final reconstruction
- Improves clarity for disease detection

### ✔ 4-Class Disease Classifier
Detects:
- Normal
- DR
- Cataract
- Glaucoma

### ✔ Visual Explanations
- Grad-CAM heatmaps for DR
- Optic Disc/Cup segmentation overlays for Glaucoma
- Blur/Opacity maps for Cataract (planned)

### ✔ Unified Interactive Detector
Input an image → System outputs:
- Final disease label
- Confidence score
- Visual explanation images

## 📂 Project Structure

```
project/
│
├── analysis/
│   └── inspect_dataset.py
│
├── datasets/
│   ├── input/
│   ├── validation/
│   └── balanced_dataset/ (Normal, DR, Cataract, Glaucoma)
│
├── models/
│   └── four_class_classifier.pth
│
├── pipeline/
│   └── multi_disease_pipeline.py
│
├── scripts/
│   ├── preprocess.py
│   ├── model_init.py
│   ├── feature_encoder.py
│   ├── multiscale_upsample.py
│   ├── feature_fusion.py
│   ├── reconstruction.py
│   ├── four_class_classifier.py
│   ├── glaucoma_module.py
│   ├── dr_classifier.py (legacy; used for DR Grad-CAM)
│   └── cataract_classifier.py (placeholder)
│
├── training/
│   └── train_4_class_classifier.py
│
├── outputs/
│   ├── enhanced/
│   └── visuals/ (Grad-CAM, OD/OC overlays)
│
└── run_disease_detector.py
```

## 🧪 Training the 4-Class Classifier

Run:

```
python project/training/train_4_class_classifier.py --epochs 10 --batch_size 32 --arch resnet18 --workers 4
```

Requirements:
- EfficientNet-B0 or ResNet18 backbone (auto-detected at inference)
- Balanced dataset in:
  `project/datasets/balanced_dataset/`

Dataset structure example:

```
balanced_dataset/
  Normal/
  DR/
  Cataract/
  Glaucoma/
```

Outputs:
- `project/models/four_class_classifier.pth`

## 🔍 Running the Interactive Disease Detector

Usage:

```
python project/run_disease_detector.py <input_image_path>
```

Example:

```
python project/run_disease_detector.py project/datasets/validation/images/0_left.jpg
```

Output includes:
- Predicted disease
- Confidence score
- Grad-CAM (if DR)
- OD/OC overlay (if Glaucoma)

## 📊 Datasets Used

1. APTOS 2019 — DR + Normal
2. ODIR — Normal, Cataract, DR, Glaucoma
3. Drishti-GS1 — OD/OC masks for Glaucoma
4. Nuclear Cataract Dataset — Cataract class

Merged & balanced dataset stored in:
`project/datasets/balanced_dataset/`

## 🏗 Model Architecture

### Enhancement Pipeline
- Feature encoder
- Multi-scale upsampling
- Residual fusion
- Final reconstruction

### Disease Classification
- EfficientNet-B0 or ResNet18
- Final layer: 4 outputs
- Softmax for probabilities

## 🖼 Visual Explanation Samples

Sample images saved to `project/outputs/visuals/`:
- DR heatmap: `<filename>_dr_heatmap.png`
- Glaucoma OD/OC overlay: `<filename>_glaucoma_overlay.png`
- Cataract blur/opacity maps (planned)

## 💻 Tech Stack

- Python
- PyTorch
- EfficientNet / ResNet
- Real-ESRGAN
- Grad-CAM
- OpenCV

## 📈 Future Enhancements

- Add AMD detection
- OCT image support
- Deploy as web app (Streamlit)
- Mobile-optimized version

## 🏆 Author

Name: Rahul
Project: OptiPredict — Retinal Disease Detection

