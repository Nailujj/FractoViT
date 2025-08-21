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
File: optim/scheduler.py

Learning-rate schedulers and warmup strategies.

Responsibilities
- Create LR schedulers (e.g., cosine annealing with warmup) as described in the paper.
- Optionally provide alternative schemes (OneCycle, StepLR, CosineWR) via config.

Inputs
- Optimizer instance and scheduler-related hyperparameters

Outputs
- Scheduler object controlling LR over steps/epochs

Project integration
- Paired with `optim.optimizer` in all pipelines; configured via `configs/*.yaml`.
"""

from timm.scheduler.cosine_lr import CosineLRScheduler

def build_scheduler(optimizer, total_steps, cfg):
    """
    Erstellt einen Cosine-LR-Scheduler mit Warmup.
    total_steps = epochs * steps_per_epoch
    """
    return CosineLRScheduler(
        optimizer,
        t_initial      = total_steps,
        warmup_t       = cfg.scheduler.warmup_steps,
        warmup_lr_init = cfg.optimizer.lr / 10,
        lr_min         = cfg.scheduler.eta_min,
    )