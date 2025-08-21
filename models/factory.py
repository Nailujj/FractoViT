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
File: models/factory.py

Central model factory for consistent model construction and configuration.

Responsibilities
- Instantiate the desired backbone (e.g., ViT-Base-Patch16-224) with pretrained weights.
- Configure classifier head, number of classes, stochastic depth, patch drop, etc.
- Optionally enable magnification-aware variants and feature exposure for XAI.

Inputs
- Model-related parameters from the config

Outputs
- Initialized `nn.Module` ready for optimization

Project integration
- Single entry point to create models consistently across train/HPO/CV scripts.
"""


import timm
import torchvision.models as tvm
from omegaconf import DictConfig
from .vit_mag import ViTWithMagnification

def get_model(name: str, num_classes: int, cfg: DictConfig = None):
    name = name.lower()
    
    # ------------------  Prototyp für magnification aware ViT. Code in vit_mag.py--------------------
    if name == "vit_base_mag": 
        
        return ViTWithMagnification(
            "vit_base_patch16_224",
            num_classes,
            mag_emb_dim = cfg.model.mag_emb_dim,
            use_imagenet_pretrain= cfg.model.use_imagenet_pretrain,
            custom_pretrained_path= cfg.model.custom_pretrained_path,           
            freeze_backbone = cfg.freeze_backbone,
        )    
    #----------------------------------------

    # timm-Pfad 
    try:
        timm_kwargs = {"num_classes": num_classes}
        if cfg is not None and name.startswith("vit"):
            timm_kwargs["drop_path_rate"]  = cfg.model.drop_path_rate
            timm_kwargs["patch_drop_rate"] = cfg.model.patch_drop_rate
        return timm.create_model(name, pretrained=True, **timm_kwargs)
    except Exception:
        pass

    #  torchvision-Fallback
    if name in tvm.__dict__:
        model_fn = tvm.__dict__[name]
        try:
            weights = tvm.__dict__[f"{name}_Weights"].IMAGENET1K_V1
            return model_fn(weights=weights, num_classes=num_classes)
        except Exception:
            return model_fn(pretrained=True, num_classes=num_classes)

    raise ValueError(f"Unbekanntes Modell: {name}")
