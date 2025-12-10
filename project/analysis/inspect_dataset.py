import os
import sys
import json
import csv
from pathlib import Path

KNOWN_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
KNOWN_LABEL_EXTS = {".csv", ".json"}

DEF_CLASSES = {"normal", "dr", "diabetic", "diabetic_retinopathy", "cataract", "glaucoma"}
MASK_HINTS = {"mask", "gt", "od_mask", "oc_mask", "opticdisc", "opticcup"}

def gather_files(root: Path):
    images = []
    masks = []
    csvs = []
    jsons = []
    folders = set()
    unknown_exts = set()

    for p in root.rglob("*"):
        if p.is_dir():
            folders.add(p.name)
            continue
        ext = p.suffix.lower()
        name_low = p.name.lower()
        if ext in KNOWN_IMAGE_EXTS:
            images.append(p)
            if any(h in name_low for h in MASK_HINTS) or any(h in str(p.parent).lower() for h in MASK_HINTS):
                masks.append(p)
        elif ext == ".csv":
            csvs.append(p)
        elif ext == ".json":
            jsons.append(p)
        else:
            if ext:
                unknown_exts.add(ext)

    return {
        "images": images,
        "masks": masks,
        "csvs": csvs,
        "jsons": jsons,
        "folders": sorted(folders),
        "unknown_exts": sorted(unknown_exts)
    }

def detect_subfolder_classes(root: Path):
    classes = set()
    for d in root.rglob("*"):
        if d.is_dir():
            n = d.name.lower()
            if n in DEF_CLASSES:
                if "diabetic" in n:
                    classes.add("DR")
                elif n == "dr":
                    classes.add("DR")
                elif n == "normal":
                    classes.add("Normal")
                elif n == "cataract":
                    classes.add("Cataract")
                elif n == "glaucoma":
                    classes.add("Glaucoma")
    return sorted(classes)

def infer_dataset_type(files, root: Path):
    has_csv = len(files["csvs"]) > 0
    has_masks = len(files["masks"]) > 0
    sub_classes = detect_subfolder_classes(root)

    if has_csv:
        try:
            for csv_path in files["csvs"]:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    cols = [c.lower() for c in (reader.fieldnames or [])]
                    if any(c in cols for c in ["category", "label", "target", "diagnosis"]):
                        return "ODIR-like (Multi-disease)"
        except Exception:
            pass
        return "ODIR-like (Multi-disease)"

    if has_masks or any(h in "/".join(files["folders"]).lower() for h in MASK_HINTS):
        return "Glaucoma Segmentation (Drishti)"

    if sub_classes:
        if set(sub_classes) == {"DR", "Normal"}:
            return "DR Dataset (APTOS-like)"
        if "Cataract" in sub_classes and ("Normal" in sub_classes or len(sub_classes) == 1):
            return "Cataract Dataset"
        if len(sub_classes) >= 3:
            return "Multi-disease Folder-structured"

    return "Unknown / Mixed Dataset"

def summarize_diseases(root: Path):
    mapping = {
        "Normal": ["normal"],
        "DR": ["dr", "diabetic", "diabetic_retinopathy"],
        "Cataract": ["cataract"],
        "Glaucoma": ["glaucoma"]
    }
    counts = {k: 0 for k in mapping.keys()}
    for dclass, keys in mapping.items():
        for key in keys:
            for img in root.rglob(f"**/{key}/*"):
                if img.is_file() and img.suffix.lower() in KNOWN_IMAGE_EXTS:
                    counts[dclass] += 1
    return counts

def main():
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        try:
            dataset_path = input("Enter dataset folder path: ").strip()
        except EOFError:
            print("❌ No input provided")
            return

    dataset_path = dataset_path.strip('"').strip("'")
    root = Path(dataset_path)
    if not root.exists() or not root.is_dir():
        print("❌ No images found — check folder path")
        return

    files = gather_files(root)
    dataset_type = infer_dataset_type(files, root)
    disease_counts = summarize_diseases(root)

    num_images = len(files["images"])
    num_masks = len(files["masks"])
    num_csv = len(files["csvs"])
    num_json = len(files["jsons"])

    print("=== DATASET INSPECTION REPORT ===")
    print(f"Dataset Path: {root}")
    print(f"Total Images: {num_images}")
    print(f"Mask Files: {num_masks}")
    print(f"CSV Files: {num_csv}")
    print(f"JSON Files: {num_json}")
    print(f"Detected Subfolders: {files['folders'][:20]}")
    print(f"Unknown File Types: {files['unknown_exts']}")
    print(f"Dataset Type: {dataset_type}")

    detected_labels = [k for k, v in disease_counts.items() if v > 0]
    if detected_labels:
        print(f"Detected Disease Labels: {detected_labels}")

    if num_masks > 0 and (num_csv + num_json) == 0:
        print("⚠ Warning: Masks present but no label files detected")
    if not detected_labels and num_csv == 0 and num_json == 0:
        print("⚠ Warning: No subfolders and no CSV/JSON labels detected")
    if len(detected_labels) == 1:
        print("⚠ Warning: Dataset contains only one class")

    missing = [k for k in ["Normal", "DR", "Cataract", "Glaucoma"] if disease_counts.get(k, 0) == 0]
    if missing:
        print(f"⚠ Warning: Missing labels for: {missing}")

    if dataset_type == "Unknown / Mixed Dataset":
        print("⚠ Unable to classify dataset structure")

    if num_images == 0:
        print("❌ No images found — check folder path")
    else:
        print("Status: READY FOR MERGING")

if __name__ == "__main__":
    main()

