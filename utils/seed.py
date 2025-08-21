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
File: utils/seed.py

Deterministic seeding and reproducibility utilities.

Responsibilities
- Seed `random`, `numpy`, and `torch` (CPU/GPU) and configure deterministic/cuDNN options.
- Provide a single function to establish reproducible runs across scripts.

Inputs
- Seed value from config or CLI

Outputs
- None (sets global state for deterministic behavior)

Project integration
- Called early in `01_train.py`, `02_hpo.py`, and `04a_crossvalidate.py`.
"""



import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
