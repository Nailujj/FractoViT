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
File: 04a_crossvalidate.py

Stratified k-fold cross-validation runner.

Responsibilities
- Create stratified folds and manage per-fold training/validation runs.
- Aggregate metrics across folds (accuracy, precision, recall, macro-F1),
  compute mean/SD/CI, and optionally produce confusion matrices.
- Ensure consistent seeding

Inputs
- Config files (`configs/*.yaml`)
- Dataset metadata/paths as configured

Outputs
- Fold-wise and aggregated results tables, confusion matrices, optional curves

Project integration
- Mirrors the evaluation protocol described in the paper; ensures robust estimates.
"""



from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

# ────────────────────────────────────────────────────────────────────────────────
# Project‑local imports (assumed to be in PYTHONPATH)
# ────────────────────────────────────────────────────────────────────────────────
from data.dataset import CustomImageDataset, build_transforms, filter_none_collate
from data.loader import build_sampler
from models.classifier import build_classifier
from optim.optimizer import build_optimizer
from optim.scheduler import build_scheduler
from utils.seed import set_seed

# ────────────────────────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────────────────────────

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: torch.Tensor, gamma: float) -> torch.Tensor:
    """Multiclass focal loss.

    Parameters
    ----------
    logits : (B, C) unnormalised model outputs
    targets : (B,) ground‑truth class indices
    alpha : per‑class weight tensor (C,)
    gamma : focusing parameter > 0
    """
    ce = torch.nn.functional.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def save_confusion_png(cm: np.ndarray, class_names: List[str], out_path: os.PathLike) -> None:
    """Plot and save a confusion‑matrix heatmap with consistent styling."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────────
# Main training routine with Hydra config
# ────────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:  # noqa: C901  # (function is long but flat)
    # ------------------------------------------------------------------
    # 0) Deterministic behaviour
    # ------------------------------------------------------------------
    set_seed(cfg.seed) # including cuda deterministic

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1) MLflow experiment init
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(cfg.logger.mlflow_experiment)

    # ------------------------------------------------------------------
    # 2) Load full dataset once to obtain labels + indices for CV split
    # ------------------------------------------------------------------
    full_ds = CustomImageDataset(
        annotations_file=cfg.dataset.annotations,
        img_dir=cfg.dataset.img_dir,
        label_map=cfg.dataset.label_map,
        transform=None,
        minority_label=None,
        minority_aug_factor=1,
        minority_tf=None,
    )
    all_labels: List[int] = full_ds.labels
    indices = list(range(len(full_ds)))

    skf = StratifiedKFold(n_splits=cfg.cv.folds, shuffle=True, random_state=cfg.seed)

    fold_results: List[Dict[str, float]] = []
    cms: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # 3) Cross‑validation outer loop
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name="cross_validation", nested=False):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, all_labels), start=1):
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.log_param("fold", fold)

                # 3.1 Transforms
                train_tf = build_transforms(cfg, "train")
                minority_tf = (
                    build_transforms(cfg, "minority") if hasattr(cfg.dataset, "minority_tf") else train_tf
                )
                val_tf = build_transforms(cfg, "val")

                # 3.2 Dataset subsets
                train_ds = CustomImageDataset(
                    annotations_file=cfg.dataset.annotations,
                    img_dir=cfg.dataset.img_dir,
                    label_map=cfg.dataset.label_map,
                    transform=train_tf,
                    minority_label=cfg.dataset.minority_label,
                    minority_aug_factor=cfg.dataset.minority_aug_factor,
                    minority_tf=minority_tf,
                )
                val_ds = CustomImageDataset(
                    annotations_file=cfg.dataset.annotations,
                    img_dir=cfg.dataset.img_dir,
                    label_map=cfg.dataset.label_map,
                    transform=val_tf,
                )

                train_sub = Subset(train_ds, train_idx)
                val_sub = Subset(val_ds, val_idx)

                # 3.3 Sampler + loaders
                train_labels = [all_labels[i] for i in train_idx]
                sampler = build_sampler(train_labels)

                num_workers = getattr(cfg, "num_workers", 4)
                train_loader = DataLoader(
                    train_sub,
                    batch_size=cfg.batch_size,
                    sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=filter_none_collate,
                )
                val_loader = DataLoader(
                    val_sub,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=filter_none_collate,
                )

                # 3.4 Model & optimisation
                n_classes = len(cfg.dataset.label_map)
                model = build_classifier(cfg.model._model_, n_classes, cfg).to(device)

                optimizer = build_optimizer(model, cfg)
                total_steps = cfg.training.epochs * len(train_loader)
                scheduler = build_scheduler(optimizer, total_steps, cfg)
                scaler = GradScaler()

                alpha = torch.ones(n_classes, device=device)
                gamma = cfg.dataset.focal_gamma
                criterion = lambda lg, lbl: focal_loss(lg, lbl, alpha, gamma)

                best_val: Dict[str, float] = {"f1": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "epoch": 0}
                patience_cnt = 0

                # 3.5 Training loop
                for epoch in range(cfg.training.epochs):
                    # ---- TRAIN ----
                    model.train()
                    train_losses = 0.0
                    train_preds: List[int] = []
                    train_trues: List[int] = []

                    for step, (imgs, labels) in enumerate(train_loader):
                        imgs, labels = imgs.to(device), labels.to(device)
                        with autocast():
                            logits = model(imgs)
                            loss = criterion(logits, labels) / cfg.grad_accum
                        scaler.scale(loss).backward()

                        train_losses += loss.item() * imgs.size(0) * cfg.grad_accum
                        train_preds.extend(logits.argmax(1).cpu().tolist())
                        train_trues.extend(labels.cpu().tolist())

                        if (step + 1) % cfg.grad_accum == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            step_idx = epoch * len(train_loader) + step
                            scheduler.step(step_idx)

                    # metrics train
                    train_loss = train_losses / len(train_sub)
                    train_acc = accuracy_score(train_trues, train_preds)
                    train_f1 = f1_score(train_trues, train_preds, average="macro", zero_division=0)
                    train_prec = precision_score(train_trues, train_preds, average="macro", zero_division=0)
                    train_rec = recall_score(train_trues, train_preds, average="macro", zero_division=0)

                    # ---- VALIDATION ----
                    model.eval()
                    val_losses = 0.0
                    val_preds: List[int] = []
                    val_trues: List[int] = []
                    with torch.no_grad():
                        for imgs, labels in val_loader:
                            imgs, labels = imgs.to(device), labels.to(device)
                            logits = model(imgs)
                            val_losses += criterion(logits, labels).item() * imgs.size(0)
                            val_preds.extend(logits.argmax(1).cpu().tolist())
                            val_trues.extend(labels.cpu().tolist())

                    val_loss = val_losses / len(val_sub)
                    val_acc = accuracy_score(val_trues, val_preds)
                    val_f1 = f1_score(val_trues, val_preds, average="macro", zero_division=0)
                    val_prec = precision_score(val_trues, val_preds, average="macro", zero_division=0)
                    val_rec = recall_score(val_trues, val_preds, average="macro", zero_division=0)

                    # ---- LOGGING (per epoch) ----
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "train_f1": train_f1,
                            "train_precision": train_prec,
                            "train_recall": train_rec,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_f1": val_f1,
                            "val_precision": val_prec,
                            "val_recall": val_rec,
                        },
                        step=epoch,
                    )

                    # ---- EARLY STOPPING DISABLED FOR CV 
                    if val_f1 > best_val["f1"]:
                        best_val.update({"f1": val_f1, "acc": val_acc, "prec": val_prec, "rec": val_rec, "epoch": epoch})
                        patience_cnt = 0
                        ckpt = Path.cwd() / f"best_model_fold_{fold}.pth"
                        torch.save(model.state_dict(), ckpt)
                        mlflow.log_artifact(str(ckpt))
                    else:
                        patience_cnt += 1
                        print("no improvement for {patience_cnt} epochs")
                        #if patience_cnt >= cfg.training.early_stop_patience:
                            #print(f"Early stop (fold {fold}, epoch {epoch})")
                        #break

                # 3.6 Evaluate best checkpoint on validation
                model.load_state_dict(torch.load(ckpt, map_location=device))
                model.eval()
                all_preds: List[int] = []
                all_trues: List[int] = []
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        logits = model(imgs)
                        all_preds.extend(logits.argmax(1).cpu().tolist())
                        all_trues.extend(labels.cpu().tolist())

                cm = confusion_matrix(all_trues, all_preds, normalize="true")
                cms.append(cm)

                # Save matrix PNG & log
                cm_png = Path.cwd() / f"cm_fold{fold}.png"
                save_confusion_png(cm, list(cfg.dataset.label_map.keys()), cm_png)
                mlflow.log_artifact(str(cm_png))

                # ---- Fold summary ----
                mlflow.log_metric("best_val_f1", best_val["f1"])
                mlflow.log_metric("best_epoch", best_val["epoch"])

                fold_results.append(
                    {
                        "fold": fold,
                        "val_f1": best_val["f1"],
                        "val_acc": best_val["acc"],
                        "val_precision": best_val["prec"],
                        "val_recall": best_val["rec"],
                    }
                )

                # Free GPU VRAM before next fold
                torch.cuda.empty_cache()
                gc.collect()

        # ------------------------------------------------------------------
        # 4) Aggregate across folds
        # ------------------------------------------------------------------
        df = pd.DataFrame(fold_results)
        csv_path = Path("cv_results_clean.csv")
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        # Weighted average confusion matrix (weight = fold val size)
        fold_sizes = [len(val_idx) for _, val_idx in skf.split(indices, all_labels)]
        avg_cm = np.average(cms, axis=0, weights=fold_sizes)
        cm_df = pd.DataFrame(avg_cm, index=cfg.dataset.label_map.keys(), columns=cfg.dataset.label_map.keys())
        cm_csv = Path("avg_confusion_matrix_clean.csv")
        cm_df.to_csv(cm_csv, index=True)
        mlflow.log_artifact(str(cm_csv))

        # Also save as PNG for poster
        avg_png = Path("avg_confusion_matrix.png")
        save_confusion_png(avg_cm, list(cfg.dataset.label_map.keys()), avg_png)
        mlflow.log_artifact(str(avg_png))


if __name__ == "__main__":
    main()
