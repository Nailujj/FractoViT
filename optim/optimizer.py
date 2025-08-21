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
File: optim/optimizer.py

Optimizer construction and parameter grouping.

Responsibilities
- Build optimizers (typically AdamW) with sensible parameter groups
  (e.g., exclude weight decay on bias/norm; layer-wise LR decay for ViT).
- Read hyperparameters from the config.

Inputs
- Model parameters and optimizer settings

Outputs
- Configured optimizer instance

Project integration
- Consumed by training/HPO/CV scripts in tandem with `optim.scheduler`.
"""

from timm.optim import create_optimizer_v2

def build_optimizer(model, cfg):
    """
    Erstellt einen AdamW-Optimizer mit Layer-Decay
    """
    # Falls unterschiedliche LR für Backbone/Head gewünscht
    # backbone_params = []
    # head_params = []
    # for name, param in model.named_parameters():
    #     if "head" in name:
    #         head_params.append(param)
    #     else:
    #         backbone_params.append(param)
    # param_groups = [
    #     {"params": backbone_params, "lr": cfg.optimizer.lr},
    #     {"params": head_params,     "lr": cfg.optimizer.head_lr}
    # ]

    # Create optimizer v2 unterstützt layer_decay direkt
    optimizer = create_optimizer_v2(
        model,
        opt=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        layer_decay=cfg.optimizer.layer_decay,
        betas=tuple(cfg.optimizer.betas) if hasattr(cfg.optimizer, 'betas') else None,
        eps=getattr(cfg.optimizer, 'eps', None)
    )
    return optimizer