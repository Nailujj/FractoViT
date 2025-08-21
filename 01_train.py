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
File: 01_train.py

Main training entry point for the ViT-based fracture-origin classifier.

Responsibilities
- Load runtime configuration (from `configs/`), set deterministic seeds.
- Construct datasets and dataloaders via `data.dataset` and `data.loader`.
- Build the model through `models.factory` (backbone, head, magnification options).
- Initialize optimizer and LR scheduler (`optim.optimizer`, `optim.scheduler`).
- Execute the training loop (epochs, logging, checkpointing, best-model tracking).
- Persist artifacts for reproducibility (metrics, configs, weights).

Inputs
- Config files in `configs/*.yaml`
- Dataset metadata/paths as specified in the config

Outputs
- Model checkpoints, training/validation metrics, and logs

Project integration
- Orchestrates all components; reference implementation for experiments reported in the paper.
"""

import gc
import os
from collections import Counter

import hydra
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast

from data.loader import build_loaders
from models.classifier import build_classifier
from optim.optimizer import build_optimizer
from optim.scheduler import build_scheduler
from utils.seed import set_seed

# timm‑Extras für Mixup & Loss
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

# ----------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------

def plot_cm(cm: np.ndarray, classes: list[str], title: str, path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks, classes, rotation=45, ha="right")
    ax.set_yticks(ticks, classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    torch.cuda.empty_cache(); gc.collect(); set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Data -----------------------------------------------------------------
    train_loader, val_loader, test_loader = build_loaders(cfg)
    n_classes = len(cfg.dataset.label_map)
    classes   = list(cfg.dataset.label_map.keys())

    # Mixup / CutMix --------------------------------------------------------
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.0,
        num_classes=n_classes,
    )
    criterion = SoftTargetCrossEntropy()

    # Model / Opt / Sched ---------------------------------------------------
    model       = build_classifier(cfg.model._model_, n_classes, cfg).to(device)
    optimizer   = build_optimizer(model, cfg)
    total_steps = cfg.training.epochs * len(train_loader)
    scheduler   = build_scheduler(optimizer, total_steps, cfg)
    scaler      = GradScaler()

    # MLflow ----------------------------------------------------------------
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(cfg.logger.mlflow_experiment)

    with mlflow.start_run(run_name=cfg.model._model_):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        best_f1, patience, global_step = 0., 0, 0

        for epoch in range(cfg.training.epochs):
            # ---------- Train --------------------------------------------
            model.train(); run_loss = 0.; y_t, y_p = [], []
            for step, (imgs, labels) in enumerate(train_loader):
                if imgs is None: continue
                imgs, labels = imgs.to(device), labels.to(device)
                # ► Mixup / CutMix
                imgs, targets = mixup_fn(imgs, labels)
                with autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, targets) / cfg.grad_accum
                scaler.scale(loss).backward()
                run_loss += loss.item() * imgs.size(0) * cfg.grad_accum

                # harte Labels für Metrik (argmax auf *targets*)
                hard_targets = targets.argmax(1)
                y_t += hard_targets.cpu().tolist(); y_p += logits.argmax(1).cpu().tolist()

                if (step + 1) % cfg.grad_accum == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                scheduler.step(global_step); global_step += 1

            tr_loss = run_loss / len(train_loader.dataset)
            tr_f1   = f1_score(y_t, y_p, average="macro", zero_division=0)
            tr_acc  = accuracy_score(y_t, y_p)

            # ---------- Validation --------------------------------------
            model.eval(); v_loss = 0.; y_tv, y_pv = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    if imgs is None: continue
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = model(imgs)
                    v_loss += criterion(logits, torch.nn.functional.one_hot(labels, n_classes).float()).item() * imgs.size(0)
                    y_tv += labels.cpu().tolist(); y_pv += logits.argmax(1).cpu().tolist()
            v_loss /= len(val_loader.dataset)

            v_f1   = f1_score(y_tv, y_pv, average="macro", zero_division=0)
            v_acc  = accuracy_score(y_tv, y_pv)
            v_prec = precision_score(y_tv, y_pv, average="macro", zero_division=0)
            v_rec  = recall_score(y_tv, y_pv, average="macro", zero_division=0)

            mlflow.log_metrics({"train_loss": tr_loss, "train_f1": tr_f1, "train_acc": tr_acc,
                                "val_loss": v_loss,   "val_f1": v_f1,   "val_acc": v_acc,
                                "val_prec": v_prec,   "val_rec": v_rec}, step=epoch)
            print(f"Epoch {epoch:02d} | train_f1={tr_f1:.3f} | val_f1={v_f1:.3f}")

            cm_val = confusion_matrix(y_tv, y_pv)
            cm_path = f"cm_val_epoch{epoch:02d}.png"; plot_cm(cm_val, classes, f"Val (E{epoch})", cm_path)
            mlflow.log_artifact(cm_path); os.remove(cm_path)

            # Early‑Stopping ---------------------------------------------
            if v_f1 > best_f1:
                best_f1, patience = v_f1, 0
                torch.save(model.state_dict(), "best_mixup.pth")
            else:
                patience += 1
                if patience >= cfg.training.early_stop_patience:
                    print("Early stopping"); break

        # ---------- Test --------------------------------------------------
        model.load_state_dict(torch.load("best_mixup.pth")); model.eval()
        y_tt, y_pt = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                if imgs is None: continue
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                y_tt += labels.cpu().tolist(); y_pt += logits.argmax(1).cpu().tolist()
        tst = {
            "test_f1": f1_score(y_tt, y_pt, average="macro", zero_division=0),
            "test_acc": accuracy_score(y_tt, y_pt),
            "test_prec": precision_score(y_tt, y_pt, average="macro", zero_division=0),
            "test_rec": recall_score(y_tt, y_pt, average="macro", zero_division=0),
        }
        mlflow.log_metrics(tst); print("Test:", tst)

        cm_test = confusion_matrix(y_tt, y_pt)
        plot_cm(cm_test, classes, "Confusion Test", "cm_test.png"); mlflow.log_artifact("cm_test.png")


if __name__ == "__main__":
    main()
