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
File: 02_hpo.py

Hyperparameter optimization (HPO) orchestrator (e.g., Optuna/TPESampler).

Responsibilities
- Define search space (LR, weight decay, drop-path, label smoothing, warmup,
  layer-wise LR decay, RandAugment ops/magnitude, patch-drop, etc.).
- Implement objective (validation macro-F1), with early pruning of poor trials.
- For each trial: build datasets/dataloaders, create model via `models.factory`,
  set up optimizer/scheduler, run a short training+validation cycle, report metrics.

Inputs
- Base config in `configs/`, plus HPO ranges (inline or via additional YAML)
- Same data interfaces as `01_train.py`

Outputs
- Best trial summary (hyperparameters + score), optional study export (JSON/CSV)

Project integration
- Produces the model configuration later used in `01_train.py` and cross-validation.
"""


from __future__ import annotations

import hydra, optuna, mlflow, torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

from data.loader import build_loaders
from models.classifier import build_classifier
from optim.optimizer import build_optimizer
from optim.scheduler import build_scheduler

from datetime import datetime

STORAGE = f"sqlite:///optuna_vit_hpo.db"

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def sample_hparams(trial, cfg):
    cfg.optimizer.lr           = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    cfg.optimizer.weight_decay = trial.suggest_float("wd", 0.0, 0.3)
    cfg.optimizer.layer_decay  = trial.suggest_float("layer_decay", 0.6, 1.0)
    # Scheduler
    sched = trial.suggest_categorical("sched", ["cosine", "onecycle"])
    if sched == "cosine":
        cfg.scheduler.name        = "cosine_with_warmup"
        cfg.scheduler.warmup_steps = trial.suggest_int("warmup_steps", 100, 800)
    else:
        cfg.scheduler.name        = "onecycle"; cfg.scheduler.warmup_steps = 0
    cfg.model.drop_path_rate  = trial.suggest_float("dpr", 0.05, 0.2)
    cfg.model.patch_drop_rate = trial.suggest_float("patch_drop", 0.0, 0.2)
    cfg.label_smoothing       = trial.suggest_float("ls", 0.0, 0.1)

    cfg.augment.mixup_alpha   = trial.suggest_float("mixup", 0.0, 0.8)
    cfg.augment.cutmix_alpha  = trial.suggest_float("cutmix", 0.0, 1.0)
    cfg.augment.mixup_prob    = trial.suggest_float("mix_prob", 0.3, 1.0)

    cfg.augment.randaugment_nops      = trial.suggest_int("ra_nops", 1, 3)
    cfg.augment.randaugment_magnitude = trial.suggest_int("ra_mag", 5, 15)

    if "loss" not in cfg:
        cfg.loss = {}
    cfg.loss.type  = trial.suggest_categorical("loss", ["ce", "focal"])
    cfg.loss.gamma = trial.suggest_float("gamma", 1.0, 3.0) if cfg.loss.type == "focal" else 0.0

    sampler = trial.suggest_categorical("sampler", ["weighted", "shuffle"])
    if "loader" not in cfg:
        cfg.loader = {}
    cfg.loader.use_sampler      = sampler == "weighted"
    cfg.dataset.minority_aug_factor = trial.suggest_int("dup", 1, 4) if not cfg.loader.use_sampler else 1


def get_loss(cfg):
    if cfg.loss.type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    class Focal(nn.Module):
        def __init__(self, gamma): super().__init__(); self.g=gamma
        def forward(self,x,y): ce=nn.functional.cross_entropy(x,y,reduction="none"); pt=torch.exp(-ce); return ((1-pt)**self.g*ce).mean()
    return Focal(cfg.loss.gamma)

# -----------------------------------------------------------------------------
# Objective function
# -----------------------------------------------------------------------------

def objective(trial, base_cfg):
    cfg = base_cfg.copy(); sample_hparams(trial, cfg)
    train_loader, val_loader, _ = build_loaders(cfg)
    n_classes = len(cfg.dataset.label_map)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model  = build_classifier(cfg.model._model_, n_classes, cfg).to(device)
    criterion = get_loss(cfg)
    optimizer = build_optimizer(model, cfg)

    # Scheduler
    if cfg.scheduler.name == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(optimizer, max_lr=cfg.optimizer.lr,
                               steps_per_epoch=len(train_loader), epochs=cfg.training.epochs, pct_start=0.1)
    else:
        scheduler = build_scheduler(optimizer, cfg.training.epochs * len(train_loader), cfg)

    mixup_fn = Mixup(mixup_alpha=cfg.augment.mixup_alpha, cutmix_alpha=cfg.augment.cutmix_alpha,
                     prob=cfg.augment.mixup_prob, switch_prob=0.5, mode='batch',
                     label_smoothing=cfg.label_smoothing, num_classes=n_classes)
    soft_crit = SoftTargetCrossEntropy(); scaler = GradScaler()
    best_f1, patience, global_step = 0., 0, 0

    for epoch in range(cfg.training.epochs):
        model.train(); y_t, y_p = [], []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, targets = mixup_fn(imgs, labels)
            with autocast():
                logits = model(imgs)
                loss = soft_crit(logits, targets)/cfg.grad_accum if mixup_fn.mix_prob>0 else criterion(logits, labels)/cfg.grad_accum
            scaler.scale(loss).backward()
            if (global_step+1)%cfg.grad_accum==0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            if hasattr(scheduler, "step_update"):
                scheduler.step_update(global_step)   # timm-Scheduler (Cosine)
            else:
                scheduler.step()                     # OneCycle & Co.
            
            global_step+=1
            y_t+=targets.argmax(1).cpu().tolist(); y_p+=logits.argmax(1).cpu().tolist()
        model.eval(); y_tv,y_pv=[],[]
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs); y_tv+=labels.cpu().tolist(); y_pv+=logits.argmax(1).cpu().tolist()
        val_f1=f1_score(y_tv,y_pv,average="macro",zero_division=0)
        trial.report(val_f1,epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        if val_f1>best_f1: best_f1,patience=val_f1,0
        else:
            patience+=1
            if patience>=base_cfg.hpo.early_stop_patience: break
    mlflow.log_metric("best_val_f1", best_f1)   # in child run
    return best_f1

# -----------------------------------------------------------------------------
# Main – Parent‑Run + Nested Trials
# -----------------------------------------------------------------------------
@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    mlflow.set_tracking_uri("databricks")
    # ─ Ensure experiment exists ─
    exp_path = cfg.logger.mlflow_experiment
    exp = mlflow.get_experiment_by_name(exp_path)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_path)
        print(f"Created new MLflow experiment at {exp_path} (id={exp_id})")
    mlflow.set_experiment(exp_path)

    # ─ Parent‑Run ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="ViT_HPO_session") as parent:
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Study -------------------------------------------------------------
        study = optuna.create_study(
            study_name=f"optuna_vit_hpo", storage=STORAGE, load_if_exists=True,
            direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=3,reduction_factor=3))

        def _obj(trial):
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                return objective(trial, cfg)

        study.optimize(_obj, n_trials=cfg.hpo.n_trials, timeout=cfg.hpo.timeout)

        # -------- Summary into Parent ------------------------------------
        mlflow.log_metric("best_overall_f1", study.best_value)
        mlflow.log_param("best_trial", study.best_trial.number)

        df = study.trials_dataframe(attrs=("number","value","params"))
        df.to_csv("optuna_results.csv", index=False); mlflow.log_artifact("optuna_results.csv")

        try:
            from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
            plot_optimization_history(study); plt.savefig("opt_history.png"); mlflow.log_artifact("opt_history.png")
            plot_param_importances(study);     plt.savefig("opt_importance.png"); mlflow.log_artifact("opt_importance.png")
        except Exception:
            pass

if __name__ == "__main__":
    main()
