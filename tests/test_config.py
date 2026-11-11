from pathlib import Path

from train.config import TrainConfig


def test_baseline_config_loads() -> None:
    cfg = TrainConfig.from_yaml("train/configs/baseline.yaml")
    assert cfg.seed == 42
    assert cfg.dataset == "flowers102"
    assert cfg.model_name == "mobilenet_v3_small"
    assert cfg.num_classes == 102
    assert cfg.epochs >= 1
    assert cfg.max_train_samples is None


def test_smoke_config_is_capped() -> None:
    cfg = TrainConfig.from_yaml("train/configs/smoke.yaml")
    assert cfg.pretrained is False
    assert cfg.max_train_samples == 8
    assert Path(cfg.output_dir).parts[0] == "outputs"
