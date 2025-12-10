import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from project.pipeline.multi_disease_pipeline import MultiDiseasePipeline
from project.scripts.four_class_classifier import CLASS_NAMES


def test_imports_and_basic_inference():
    pipeline = MultiDiseasePipeline()
    val_dir = Path("project/datasets/validation/images")
    images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    if not images:
        print("X No validation images found — skipping integrity inference test")
        return
    img_path = images[0]
    result = pipeline.analyze(img_path)
    assert "enhanced_image" in result
    assert "final_disease_prediction" in result
    assert result["final_disease_prediction"] in CLASS_NAMES
    assert "four_class_probabilities" in result
    probs = result["four_class_probabilities"]
    assert isinstance(probs, list) and len(probs) == 4
    s = sum(probs)
    assert 0.99 <= s <= 1.01
    assert "glaucoma_status" in result
    assert "cdr_value" in result


if __name__ == "__main__":
    test_imports_and_basic_inference()
