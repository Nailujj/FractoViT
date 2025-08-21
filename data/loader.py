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

"""
File: data/loader.py

DataLoader utilities and sampling strategies.

Responsibilities
- Construct `torch.utils.data.DataLoader` for train/val.
- Configure `WeightedRandomSampler` or equivalent to mitigate class imbalance.
- Set batch size, num_workers, pin_memory, and custom collate functions if needed.

Inputs
- Dataset instances from `data.dataset`
- Loader-related parameters from config

Outputs
- DataLoader objects for training and validation

Project integration
- Utility layer consumed by `01_train.py`, `02_hpo.py`, and `04a_crossvalidate.py`.
"""


from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from collections import Counter
from .dataset import CustomImageDataset, build_splits, build_transforms, filter_none_collate


def build_sampler(labels):
    """
    Erstellt einen WeightedRandomSampler, der seltene Klassen häufiger zieht.
    labels: List[int]
    """
    counts = Counter(labels)
    weights = [1.0 / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_loaders(cfg, fixed_splits=None):

    # 1) Voll-Dataset (ohne Transform) zum Splitten
    full_dataset = CustomImageDataset(
        cfg.dataset.annotations,
        cfg.dataset.img_dir,
        cfg.dataset.label_map,
        transform=None,
        minority_label=None,
        minority_aug_factor=1,
        minority_tf=None
    )
    all_labels = full_dataset.labels
    all_indices = list(range(len(full_dataset)))

    # 2) Stratified Splits
    if fixed_splits is None:
        tr_idx, va_idx, te_idx = build_splits(
            all_indices,
            all_labels,
            val_ratio=cfg.dataset.val_ratio if hasattr(cfg.dataset, 'val_ratio') else 0.15,
            test_ratio=cfg.dataset.test_ratio if hasattr(cfg.dataset, 'test_ratio') else 0.15,
            seed=cfg.seed
        )
    else:
        tr_idx, va_idx, te_idx = fixed_splits


    # 3) Dataset-Instanzen mit Transforms + Oversampling für Train
    train_ds = CustomImageDataset(
        cfg.dataset.annotations,
        cfg.dataset.img_dir,
        cfg.dataset.label_map,
        transform=build_transforms(cfg, 'train'),
        minority_label=cfg.dataset.minority_label,
        minority_aug_factor=cfg.dataset.minority_aug_factor,
        minority_tf=build_transforms(cfg, 'minority') if hasattr(cfg.dataset, 'minority_tf') else build_transforms(cfg, 'train')
    )
    val_ds = CustomImageDataset(
        cfg.dataset.annotations,
        cfg.dataset.img_dir,
        cfg.dataset.label_map,
        transform=build_transforms(cfg, 'val')
    )
    test_ds = CustomImageDataset(
        cfg.dataset.annotations,
        cfg.dataset.img_dir,
        cfg.dataset.label_map,
        transform=build_transforms(cfg, 'val')
    )

    # 4) Subsets
    train_ds = Subset(train_ds, tr_idx)
    val_ds = Subset(val_ds, va_idx)
    test_ds = Subset(test_ds, te_idx)

    # 5) Sampler für Train
    train_labels = [all_labels[i] for i in tr_idx]
    sampler = build_sampler(train_labels)

    # 6) DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        #shuffle=True,
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True,
        collate_fn=filter_none_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True,
        collate_fn=filter_none_collate
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True,
        collate_fn=filter_none_collate
    )

    return train_loader, val_loader, test_loader
