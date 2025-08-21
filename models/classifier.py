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
File: models/classifier.py
====================

Convenience‑Function `build_classifier`, which internally delegates
`models.factory.get_model`.  Abstraction layer, to register
future Modell-Variants centralized. Currently redundant.
"""

import torch
import torch.nn as nn
import timm
from omegaconf import DictConfig
from models.factory import get_model



def build_classifier(name: str, num_classes: int, cfg: DictConfig): # unnötig
    return get_model(name, num_classes, cfg)