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
File: models/vit_mag.py

Experimental Vision Transformer backbone with optional magnification awareness.

Responsibilities
- Implement a ViT-based feature extractor with hooks for magnification features/tokens.
- Support fine-tuning options (layer-wise LR decay, stochastic depth, patch drop).
- Expose intermediate representations if required for Grad-CAM or analysis.

Inputs
- Image tensors (and optional magnification encodings)

Outputs
- Feature maps / logits feeding into the classification head

Project integration
- Used by `models.factory` and wrapped by `models.classifier`.
"""



import torch, torch.nn as nn, timm

class ViTWithMagnification(nn.Module):
    def __init__(self,
                 vit_name: str,
                 num_classes: int,
                 mag_emb_dim: int = 64,
                 use_imagenet_pretrain: bool = True,
                 custom_pretrained_path: str | None = None,
                 freeze_backbone: bool = False,
                 partial_unfreeze: bool = True,
                 dropout_p: float = 0.3):
        super().__init__()

        # ---------------- Backbone ----------------
        self.vit = timm.create_model(
            vit_name, pretrained=use_imagenet_pretrain, num_classes=0
        )

        if custom_pretrained_path:                         # optional FT-Checkpoint
            ckpt = torch.load(custom_pretrained_path, map_location="cpu")
            sd = {k.replace("module.", ""): v
                  for k,v in ckpt.get("state_dict", ckpt).items()
                  if k.replace("module.", "") in self.vit.state_dict()
                  and v.shape == self.vit.state_dict()[k.replace("module.", "")].shape}
            self.vit.load_state_dict(sd, strict=False)

        if freeze_backbone:
            for p in self.vit.parameters(): p.requires_grad = False
        if partial_unfreeze:                               # Blöcke 10-11 + End-Norm
            for n,p in self.vit.named_parameters():
                if any(s in n for s in ["blocks.10", "blocks.11", "norm"]):
                    p.requires_grad = True

        vit_dim = self.vit.num_features                    # 768  (ViT-Base)

        # ---------------- Magnification-MLP  (1 → 32 → mag_emb_dim) ---------------
        self.mag_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, mag_emb_dim),
            nn.GELU()
        )

        # ---------------- Head-MLP  (832 → Hidden → classes) ----------------
        self.dropout = nn.Dropout(dropout_p)
        hidden1, hidden2, hidden3 = 512, 256, 128

        self.head = nn.Sequential(
            nn.Linear(vit_dim + mag_emb_dim, hidden1),
            nn.GELU(),
            nn.LayerNorm(hidden1),
            nn.Dropout(dropout_p),

            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.LayerNorm(hidden2),
            nn.Dropout(dropout_p),

            nn.Linear(hidden2, hidden3),
            nn.GELU(),
            nn.LayerNorm(hidden3),
            nn.Dropout(dropout_p),

            nn.Linear(hidden3, num_classes)
        )


    # ---------------- forward --------------------------------------------
    def forward(self, x, mag):
        vit_feat = self.vit(x)                              # [B, 768]
        if mag.dim() == 1: mag = mag.unsqueeze(-1)          # [B] → [B,1]
        mag_feat = self.mag_proj(mag)                       # [B, 64]
        feat = torch.cat([vit_feat, mag_feat], dim=1)       # [B, 832]
        return self.head(self.dropout(feat))
