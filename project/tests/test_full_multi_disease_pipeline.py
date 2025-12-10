import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from project.pipeline.multi_disease_pipeline import MultiDiseasePipeline
from project.scripts.four_class_classifier import CLASS_NAMES

def test_4class_output_shapes():
    pipeline = MultiDiseasePipeline()
    val_dir = Path("project/datasets/validation/images")
    images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    if not images:
        print("X No validation images found — skipping shape test")
        return
    img_path = images[0]
    result = pipeline.analyze(img_path)
    assert result["final_disease_prediction"] in CLASS_NAMES
    assert result["four_class_prediction"] in CLASS_NAMES
    assert len(result["four_class_probabilities"]) == 4

def test_pipeline_integration():
    pipeline = MultiDiseasePipeline()
    val_dir = Path("project/datasets/validation/images")
    images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    if len(images) == 0:
        print("X No validation images found — skipping integration test")
        return
    samples_to_test = images[:3]
    for img_path in samples_to_test:
        result = pipeline.analyze(img_path)
        assert result["final_disease_prediction"] == result["four_class_prediction"]
        assert result["final_disease_prediction"] in CLASS_NAMES

if __name__ == "__main__":
    test_4class_output_shapes()
    test_pipeline_integration()
