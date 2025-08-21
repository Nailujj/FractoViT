# FractoVit: Vision Transformer-based Fracture Origin Classification

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the official implementation of the methods described in our paper:

> **insertname**
>
> Julian Schmid, et al. (2025)

---

## Overview

Ceramic materials such as BIOLOX®delta are widely used in orthopedic implants due to their outstanding wear resistance and biocompatibility.

Understanding fracture origins is critical for quality assurance, yet manual fractography is subjective and time-consuming.

This project introduces  **FractoVit** , a Vision Transformer (ViT)-based workflow for automated fracture-origin classification from scanning electron microscopy (SEM) images across extreme magnification levels.

The method achieves robust performance even at low magnification, highlighting the predictive power of macro-scale fracture patterns that human fractographers often overlook.

---

## Repository structure

```
Code_BA/
│
├── 01_train.py           # Main training pipeline
├── 02_hpo.py             # Hyperparameter optimization
├── 04a_crossvalidate.py  # k-fold cross-validation
│
├── data/                 # Dataset and dataloader definitions
│   ├── dataset.py
│   └── loader.py
│
├── models/               # Model definitions, partially experimental and not relevant to the paper
│   ├── classifier.py
│   ├── factory.py
│   └── vit_mag.py
│
├── optim/                # Optimizer & scheduler utilities
│   ├── optimizer.py
│   └── scheduler.py
│
├── utils/
│   └── seed.py           # Seeding utilities
│
├── configs/              # YAML configs for experiments
├── requirements.txt      # Python dependencies
└── LICENSE               # Apache 2.0 license
```

---

## Installation

Tested with **Python 3.10+** and  **PyTorch 2.2+** .

```bash
git clone https://github.com/your-org/fractovit.git
cd fractovit
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

---

## Data

**Note:** The SEM dataset used in this project is proprietary and cannot be published due to confidentiality.

Therefore, this repository only provides the **training/validation code** and  **configuration files** , not the raw data.

To reproduce results on your own dataset, adapt the CSV metadata format in `data/dataset.py`.

---

Results

Key findings (see paper for full details):

* Robust classification performance across magnifications (50×–10k×).
* Strong F1 scores even at low magnification (see manuscript).
* Automated workflows reduce the subjectivity and workload of expert fractography.

---

## Citation

If you use this code in your research, please cite:

```bibtex
CHANGEME
```

---

## License

This project is licensed under the **Apache License 2.0** – see [LICENSE](./LICENSE) for details.

---

## Acknowledgements

* CeramTec GmbH for providing BIOLOX®delta materials and domain expertise.
* Open-source community for PyTorch, timm, and Optuna libraries.
* Daniel Behr from the University of Applied Sciences Northwestern Switzerland (FHNW) for his feedback on the approach of this work.
