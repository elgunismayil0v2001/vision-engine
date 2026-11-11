from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from train.config import TrainConfig

# MobileNet-v3 ImageNet preprocessing. Must match export / C++ later (parity).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _subset(dataset, max_samples: int | None) -> Subset | object:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _synthetic_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, int]:
    """Deterministic fake data so tests do not download Flowers102."""
    n_train = cfg.max_train_samples or 8
    n_val = cfg.max_val_samples or 8
    g = torch.Generator().manual_seed(cfg.seed)
    train_x = torch.rand(n_train, 3, cfg.image_size, cfg.image_size, generator=g)
    train_y = torch.randint(0, cfg.num_classes, (n_train,), generator=g)
    val_x = torch.rand(n_val, 3, cfg.image_size, cfg.image_size, generator=g)
    val_y = torch.randint(0, cfg.num_classes, (n_val,), generator=g)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, n_train


def build_dataloaders(
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, int]:
    if cfg.dataset == "synthetic":
        return _synthetic_loaders(cfg)
    if cfg.dataset != "flowers102":
        raise ValueError(f"unsupported dataset {cfg.dataset!r}; week 1 uses flowers102")

    train_ds = datasets.Flowers102(
        root=cfg.data_root,
        split="train",
        download=True,
        transform=build_transforms(cfg.image_size, train=True),
    )
    val_ds = datasets.Flowers102(
        root=cfg.data_root,
        split="val",
        download=True,
        transform=build_transforms(cfg.image_size, train=False),
    )
    train_ds = _subset(train_ds, cfg.max_train_samples)
    val_ds = _subset(val_ds, cfg.max_val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, len(train_ds)
