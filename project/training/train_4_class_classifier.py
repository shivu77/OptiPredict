import os
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DATASET_DIR = Path("project/datasets/balanced_dataset")
FALLBACK_DATASET_ABS = Path(r"C:\Users\Rahul\OneDrive\Desktop\Optic\project\datasets\balanced_dataset")
MODEL_DIR = Path("project/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = MODEL_DIR / "four_class_classifier.pth"

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

def accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    return (preds == targets).float().mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--arch", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet18"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()
    dataset_dir = Path(args.data_root) if args.data_root else DATASET_DIR
    if not dataset_dir.exists() and FALLBACK_DATASET_ABS.exists():
        dataset_dir = FALLBACK_DATASET_ABS
    if not dataset_dir.exists():
        print("X Balanced dataset not found at", dataset_dir)
        return
    base_dataset = datasets.ImageFolder(dataset_dir)
    if len(base_dataset.classes) < 4:
        print("X Balanced dataset missing classes", base_dataset.classes)
        return
    n = len(base_dataset)
    val_n = max(1, int(0.2 * n))
    train_n = n - val_n
    indices = torch.randperm(n).tolist()
    train_indices = indices[:train_n]
    val_indices = indices[train_n:]
    train_dataset = datasets.ImageFolder(dataset_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(dataset_dir, transform=val_tf)
    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        total_acc = 0.0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                total_acc += accuracy(out, y)
                count += 1
        avg_loss = total_loss / max(1, len(train_loader))
        avg_acc = total_acc / max(1, count)
        print(f"Epoch {epoch}/{epochs} — Train Loss {avg_loss:.4f} — Val Acc {avg_acc:.4f}")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("✓ 4-Class Classifier Training Completed")
    print("Weights:", WEIGHTS_PATH)

if __name__ == "__main__":
    main()
