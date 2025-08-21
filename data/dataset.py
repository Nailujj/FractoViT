# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Julian Schmid and CeramTec GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# data/dataset.py
"""
File: data/dataset.py

Dataset definitions and transform pipeline for SEM micrographs.

Responsibilities
- Load image paths, labels, and optional magnification metadata from config/CSV.
- Apply training/validation transforms (e.g., RandAugment, resize, normalization).
- Currently in development: encode magnification as an additional feature/token for the model, functionality commented out.
- Return tensors with labels (and auxiliary info like magnification/ID if needed).

Inputs
- Metadata/CSV and image root(s) specified in the config

Outputs
- PyTorch-compatible samples for training and validation loops

Project integration
- Consumed by `data.loader` to build DataLoader instances used in all pipelines.
"""


import os, ast, pandas as pd, numpy as np, torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment
from timm.data import resolve_data_config
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import resize
import timm   # nur für resolve_data_config

# ------------------------------------------------------------------
# Hilfs-Klassen und -Funktionen
# ------------------------------------------------------------------
class CropBottom:
    """schneidet die unteren X % des Bildes ab (default 9 %)."""
    def __init__(self, percentage: float = 0.09):
        self.percentage = percentage
    def __call__(self, img):
        w, h = img.size
        new_h = int(h * (1 - self.percentage))
        return img.crop((0, 0, w, new_h))

def filter_none_collate(batch):
    """Entfernt None-Einträge; Standard-collate auf Rest."""
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

# ------------------------------------------------------------------
# Stratified Split
# ------------------------------------------------------------------
def build_splits(indices, labels, val_ratio=0.15, test_ratio=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    train_idx, tmp_idx = train_test_split(
        indices, test_size=val_ratio + test_ratio,
        stratify=labels, random_state=seed)
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=tmp_labels,
        random_state=seed)
    return train_idx, val_idx, test_idx

# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
def build_transforms(cfg, split: str):
    name, img_size = cfg.model._model_.lower(), cfg.img_size
    try:
        stub = timm.create_model(name, pretrained=True, num_classes=1)
        data_cfg = resolve_data_config(model=stub)
        mean, std = data_cfg["mean"], data_cfg["std"]
    except Exception:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    tf = [CropBottom(), transforms.Resize((img_size, img_size))]
    if split == "train":
        tf.append(RandAugment(
            num_ops   = cfg.augment.randaugment_nops,
            magnitude = cfg.augment.randaugment_magnitude))
    tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(tf)

# ------------------------------------------------------------------
#  CustomImageDataset
# ------------------------------------------------------------------
class CustomImageDataset(Dataset):
    """
    Mag Option derzeit nicht verwendet, da keine Verbesserung im Modell. Sonst am Ende wieder "einkommentieren"
    `mag` wird nach `mag_norm` transformiert:
        • "raw"    : unverändert
        • "log"    : log10(x)
        • "zscore" : (x-mean)/std
        • "minmax" : (x-min)/(max-min)
    """
    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        label_map: dict,
        transform=None,
        minority_label: int | None = None,
        minority_aug_factor: int = 1,
        minority_tf=None,
        mag_norm: str = "raw",          #  <<<  NEU
    ):
        self.df = pd.read_csv(annotations_file)
        self.img_dir      = img_dir
        self.transform    = transform
        self.label_map    = label_map
        self.minority_label      = minority_label
        self.minority_aug_factor = minority_aug_factor
        self.minority_tf  = minority_tf or transform
        self.mag_norm     = mag_norm.lower()

        # Daten einlesen ---------------------------------------------------
        self.paths, self.labels, self.mags_raw = self._parse_df()

        # Statistiken für z-Score / Min-Max -------------------------------
        mags_np       = np.array(self.mags_raw, dtype=np.float32)
        self.mag_mean = mags_np.mean() if mags_np.size else 0.0
        self.mag_std  = mags_np.std()  if mags_np.size else 1.0
        self.mag_min  = mags_np.min()  if mags_np.size else 0.0
        self.mag_max  = mags_np.max()  if mags_np.size else 1.0

    # ------------------------------------------------------------------
    def _parse_df(self):
        p, y, m = [], [], []
        for _, row in self.df.iterrows():
            fn   = str(row["filename"]).strip()
            frac = row["fractureType"]
            mag  = row["magnification_value"]
            if frac not in self.label_map or pd.isna(mag):
                continue
            lbl = self.label_map[frac]
            full_path = os.path.join("/dbfs", self.img_dir.replace("dbfs:/", ""), fn)
            repeat = self.minority_aug_factor if lbl == self.minority_label else 1
            for _ in range(repeat):
                p.append(full_path)
                y.append(lbl)
                m.append(float(mag))
        return p, y, m

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.paths)

    # ------------------------------------------------------------------
    def _norm_mag(self, x: float) -> float:
        if self.mag_norm == "log":
            return np.log10(x + 1e-8)
        if self.mag_norm == "zscore":
            return (x - self.mag_mean) / (self.mag_std + 1e-8)
        if self.mag_norm == "minmax":
            return (x - self.mag_min) / (self.mag_max - self.mag_min + 1e-8)
        return x   # "raw"

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        path   = self.paths[idx]
        label  = self.labels[idx]
        mag_val_raw = self.mags_raw[idx]

        # ---------- Bild laden ----------------------------------------
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return None

        # ---------- Transforms ---------------------------------------
        if label == self.minority_label:
            img = self.minority_tf(img)
        else:
            img = self.transform(img)

        # ---------- Magnification normalisieren ----------------------
        mag_val = self._norm_mag(mag_val_raw)
        mag     = torch.tensor([mag_val], dtype=torch.float32)

        return img, label
        # return img, mag, label
