import json
from pathlib import Path

from train.config import TrainConfig
from train.train import run


def test_one_epoch_on_synthetic_data(tmp_path: Path) -> None:
    cfg = TrainConfig(
        seed=7,
        dataset="synthetic",
        data_root=str(tmp_path / "data"),
        model_name="mobilenet_v3_small",
        pretrained=False,
        num_classes=5,
        image_size=32,
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
        num_workers=0,
        output_dir=str(tmp_path / "out"),
        max_train_samples=8,
        max_val_samples=8,
    )
    summary = run(cfg)

    assert (tmp_path / "out" / "last.pt").is_file()
    assert (tmp_path / "out" / "best.pt").is_file()
    assert (tmp_path / "out" / "metrics.jsonl").is_file()
    assert (tmp_path / "out" / "run.json").is_file()
    assert summary["seed"] == 7
    assert "git_hash" in summary
    assert 0.0 <= summary["best_val_acc"] <= 1.0

    rows = (tmp_path / "out" / "metrics.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["epoch"] == 1
    assert row["seed"] == 7
    assert "git_hash" in row
