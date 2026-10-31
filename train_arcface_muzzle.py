#!/usr/bin/env python3
"""
train_arcface_muzzle.py
=======================
Fine-Tuning Script for Cattle Muzzle ArcFace Embedding Matcher.

Features:
- Backbones: ConvNeXt-Tiny & ResNet-50 with GeM Pooling (p=3.0)
- ArcFace Loss: Additive Angular Margin Penalty (s=30.0, m=0.50)
- Supports:
    * Supervised cattle ID dataset (folder-per-animal)
    * Output video folder self-supervised / pseudo-ID dataset (e.g. output/video_five_frames/*)
- Heavy biometric augmentations:
    * Random rotation (-15 deg to +15 deg)
    * Random perspective warp
    * Color/Illumination jitter (contrast, brightness)
    * Random erasing
- Saves checkpoints to weights/arcface_{backbone}.pth
"""

import os
import sys
import glob
import time
import math
import random
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dl_muzzle_embedder import (
    MuzzleBiometricModel,
    ArcFaceHead,
    prepare_input_tensor,
    WEIGHTS_DIR,
)

logger = logging.getLogger("train_arcface_muzzle")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] [TrainArcFace] [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Biometric Augmentation Pipeline
# ---------------------------------------------------------------------------
class BiometricAugmentation:
    """
    Biometric transforms specialized for cattle muzzle ridge & groove maps:
    - Random rotation (+-15 deg) preserving ridge orientation
    - Random perspective warp (+-10%) simulating head tilt
    - Illumination & contrast jitter
    - Random erasing (occlusion simulation)
    """
    def __init__(self, rotation_deg: float = 15.0, p_erase: float = 0.2):
        self.rotation_deg = rotation_deg
        self.p_erase = p_erase

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is (3, H, W)
        # 1. Random rotation
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        tensor = TF.rotate(tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)

        # 2. Random horizontal flip (cattle muzzles have bilateral features)
        if random.random() > 0.5:
            tensor = TF.hflip(tensor)

        # 3. Random perspective
        if random.random() > 0.5:
            width, height = tensor.shape[-1], tensor.shape[-2]
            startpoints, endpoints = T.RandomPerspective.get_params(width, height, 0.15)
            tensor = TF.perspective(tensor, startpoints, endpoints, interpolation=TF.InterpolationMode.BILINEAR)

        # 4. Illumination / contrast jitter on channel 0 & 1
        if random.random() > 0.4:
            factor = random.uniform(0.85, 1.15)
            tensor[:2] = torch.clamp(tensor[:2] * factor, -3.0, 3.0)

        # 5. Random erasing
        if random.random() < self.p_erase:
            h, w = tensor.shape[-2], tensor.shape[-1]
            eh, ew = int(h * random.uniform(0.05, 0.15)), int(w * random.uniform(0.05, 0.15))
            ey = random.randint(0, h - eh)
            ex = random.randint(0, w - ew)
            tensor[:, ey:ey+eh, ex:ex+ew] = 0.0

        return tensor


# ---------------------------------------------------------------------------
# Cattle Muzzle Dataset
# ---------------------------------------------------------------------------
class CattleMuzzleDataset(Dataset):
    """
    Dataset loader for muzzle biometric training:
    Discovers frame sets from:
    1. Standard class folders: <data_dir>/<cattle_id>/*.jpg
    2. Extraction cache: output/video_five_frames/<video_name>/*
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        input_mode: str = "composite_3ch",
        target_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.input_mode = input_mode
        self.target_size = target_size
        self.augment = augment
        self.aug_pipe = BiometricAugmentation() if augment else None

        self.samples: List[Tuple[Dict[str, str], int]] = []
        self.classes: List[str] = []
        self._index_dataset()

    def _index_dataset(self):
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return

        # Find candidate directories
        subdirs = [p for p in self.data_dir.iterdir() if p.is_dir()]
        subdirs.sort()

        for class_idx, sdir in enumerate(subdirs):
            # Check for 5-frame extraction cache pattern
            frame_skels = sorted(list(sdir.glob("frame_*_05_thin_skeleton.png")))
            if frame_skels:
                self.classes.append(sdir.name)
                for skel_path in frame_skels:
                    stem_prefix = skel_path.name.split("_05_thin_skeleton")[0]
                    sample_dict = {
                        "skel": str(skel_path),
                        "clahe": str(sdir / f"{stem_prefix}_02_enhanced_ridge.jpg"),
                        "groove": str(sdir / f"{stem_prefix}_03_groove_response.png"),
                        "crop": str(sdir / f"{stem_prefix}_01_native_crop.jpg"),
                    }
                    self.samples.append((sample_dict, len(self.classes) - 1))
            else:
                # Direct images in subfolder
                images = sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")))
                if len(images) >= 1:
                    self.classes.append(sdir.name)
                    for img_path in images:
                        sample_dict = {
                            "crop": str(img_path),
                            "clahe": str(img_path),
                            "groove": str(img_path),
                            "skel": str(img_path),
                        }
                        self.samples.append((sample_dict, len(self.classes) - 1))

        logger.info(f"Indexed {len(self.samples)} muzzle samples across {len(self.classes)} identity classes from {self.data_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_dict, label = self.samples[idx]

        # Read images
        clahe_img = cv2.imread(sample_dict["clahe"], cv2.IMREAD_GRAYSCALE) if os.path.exists(sample_dict["clahe"]) else None
        groove_img = cv2.imread(sample_dict["groove"], cv2.IMREAD_GRAYSCALE) if os.path.exists(sample_dict["groove"]) else None
        skel_img = cv2.imread(sample_dict["skel"], cv2.IMREAD_GRAYSCALE) if os.path.exists(sample_dict["skel"]) else None

        # Fallback if any is missing
        if clahe_img is None:
            clahe_img = np.zeros(self.target_size, dtype=np.uint8)
        if groove_img is None:
            groove_img = clahe_img
        if skel_img is None:
            skel_img = clahe_img

        tensor = prepare_input_tensor(
            clahe_gray=clahe_img,
            groove_response=groove_img,
            skeleton=skel_img,
            mask=None,
            input_mode=self.input_mode,
            target_size=self.target_size,
            device="cpu",
        ).squeeze(0)  # (3, H, W)

        if self.augment and self.aug_pipe:
            tensor = self.aug_pipe(tensor)

        return tensor, label


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_arcface(
    data_dir: str,
    backbone: str = "convnext_tiny",
    input_mode: str = "composite_3ch",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "auto",
    save_path: Optional[str] = None,
):
    if str(device).lower() in ("gpu", "cuda"):
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = "cpu"
    dev = torch.device(dev_str)
    logger.info(f"Training on device: {dev_str.upper()} | Backbone: {backbone} | Mode: {input_mode}")

    dataset = CattleMuzzleDataset(data_dir=data_dir, input_mode=input_mode, augment=True)
    if len(dataset) < 2 or len(dataset.classes) < 2:
        logger.error(f"Insufficient training data in {data_dir} ({len(dataset)} samples, {len(dataset.classes)} classes). Needs >= 2 classes.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) > batch_size)

    num_classes = len(dataset.classes)
    model = MuzzleBiometricModel(backbone=backbone, pretrained=True, target_dim=512, num_classes=num_classes)
    model.to(dev)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_file = save_path or str(WEIGHTS_DIR / f"arcface_{backbone.lower()}.pth")

    logger.info(f"Starting ArcFace training for {epochs} epochs (Classes: {num_classes}, Steps/epoch: {len(dataloader)})...")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        t_start = time.time()

        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(dev), targets.to(dev)

            optimizer.zero_grad()
            logits, embs = model(inputs, labels=targets)
            loss = criterion(logits, targets)
            loss.backward()

            # Clip gradients to prevent explosion with angular margin
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

        scheduler.step()
        epoch_loss = running_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        elapsed = time.time() - t_start

        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}% | Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save trained checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "backbone": backbone,
        "input_mode": input_mode,
        "num_classes": num_classes,
        "classes": dataset.classes,
        "target_dim": 512,
    }, out_file)

    logger.info(f"Successfully trained and saved ArcFace weights to: {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Fine-Tune Cattle Muzzle ArcFace Model")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "output" / "video_five_frames"),
                        help="Path to folder containing extracted frame subdirectories or class folders")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", choices=["convnext_tiny", "resnet50"],
                        help="CNN / ConvNeXt backbone")
    parser.add_argument("--input-mode", type=str, default="composite_3ch", choices=["composite_3ch", "skeleton_only", "groove_map"],
                        help="Input tensor construction mode")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"], help="Compute device")
    parser.add_argument("--save-path", type=str, default=None, help="Custom destination path for saved .pth weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_arcface(
        data_dir=args.data_dir,
        backbone=args.backbone,
        input_mode=args.input_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path,
    )
