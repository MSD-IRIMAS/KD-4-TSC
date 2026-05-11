# KD4TSC: Knowledge Distillation for Time Series Classification

A PyTorch framework for compressing deep time series classification models via **knowledge distillation (KD)**. The project trains a large *teacher* network on the UCR Archive and then transfers its knowledge to a smaller *student* network, allowing the student to approach teacher accuracy with a fraction of the parameters.

Three backbone architectures are supported out of the box: **Inception**, **FCN**, and **ConvTran**.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architectures](#architectures)
- [Knowledge Distillation](#knowledge-distillation)
- [Output Structure](#output-structure)
- [Command-Line Arguments](#command-line-arguments)
- [Extending the Framework](#extending-the-framework)

---

## Overview

Knowledge distillation trains a compact *student* model to mimic a larger pre-trained *teacher* using a weighted combination of:

1. **Student cross-entropy loss** against ground-truth labels
2. **Distillation loss** (KL divergence) between softened teacher and student logits at a chosen temperature `T`

The combined objective is:

```
L = α · CE(student, labels) + (1 − α) · KL(softmax(student/T), softmax(teacher/T))
```

This framework runs three classifier modes:

- `teacher` — train the full-capacity model
- `student_alone` — train the compact model from scratch (KD baseline)
- `student_kd` — train the compact model with the teacher's guidance

Each experiment is repeated over multiple iterations, and the best teacher (lowest training loss) is selected to supervise the students.

---

## Repository Structure

```
KD4TSC/
├── config.py                       # All hyperparameters and paths
├── main.py                         # Experiment orchestrator
├── distiller.py                    # DistillationLoss + Distiller trainer
├── utils.py                        # Metrics, logging, teacher selection
├── analyze_results.py              # (placeholder for result analysis)
├── data/
│   └── data_utils.py               # UCR loader, znormalization, DataLoaders
├── models/
│   ├── inception.py                # InceptionTime architecture
│   ├── fcn.py                      # Fully Convolutional Network
│   ├── convtran.py                 # ConvTran (Conv + Transformer)
│   ├── Attention.py                # Self-attention variants (abs, rel-scalar, rel-vector)
│   └── AbsolutePositionalEncoding.py  # tAPE / APE / learnable positional encodings
└── trainers/
    └── trainer.py                  # Unified Trainer for all architectures
```

---

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10 (with CUDA recommended)
- NumPy, pandas, scikit-learn
- einops (used by attention modules)

### Setup

```bash
# Clone the project
git clone <your-repo-url> KD4TSC
cd KD4TSC

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy pandas scikit-learn einops
```

Verify GPU availability:

```python
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset Preparation

The framework uses the **UCR Archive 2018** (univariate time series). Each dataset is expected at:

```
<PATH_DATA>/<DatasetName>/<DatasetName>_TRAIN.tsv
<PATH_DATA>/<DatasetName>/<DatasetName>_TEST.tsv
```

Each `.tsv` file contains one sample per row; the first column is the class label and the remaining columns are the time series values.

Update `PATH_DATA` in `config.py` to point to your local UCR Archive root:

```python
PATH_DATA = '/path/to/UCRArchive_2018/'
```

---

## Quick Start

### 1. Configure the experiment

Open `config.py` and set:

- `ARCHITECTURE` → `'inception'`, `'fcn'`, or `'convtran'`
- `CLASSIFIERS` → which modes to run (e.g. `['teacher', 'student_kd', 'student_alone']`)
- `UNIVARIATE_DATASET_NAMES_2018` → list of datasets (defaults to one dataset for a quick test)
- `PATH_DATA` and `PATH_OUT` → input/output paths

### 2. Train the teacher first

Students need a trained teacher. Run only the teacher mode:

```bash
python main.py --classifiers teacher
```

This will train teachers for every dataset × iteration combination and save checkpoints under `PATH_OUT/<ARCH>/results/teacher/...`.

### 3. Train students with distillation

Once teachers exist:

```bash
python main.py --classifiers student_kd student_alone
```

The framework will automatically pick the best teacher per dataset (lowest training loss) and train students for each `(alpha, temperature)` combination defined in `config.py`.

### 4. Run a single dataset

```bash
python main.py --datasets ACSF1 Coffee --classifiers student_kd --iterations 3
```

---

## Configuration

All knobs live in `config.py`. Key sections:

### Training

```python
EPOCHS = 1500
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 50          # ReduceLROnPlateau patience
MIN_LR = 0.0001
LR_FACTOR = 0.5
```

### Knowledge distillation

```python
ALPHA_LIST = [0.5]        # weight for student CE loss
TEMPERATURE_LIST = [10]   # softening temperature
```

A full grid is run over `ALPHA_LIST × TEMPERATURE_LIST × ITERATIONS`.

### Iterations

```python
ITERATIONS = {
    'teacher': 5,
    'student_kd': 5,
    'student_alone': 5,
}
```

### Architecture-specific hyperparameters

See the [Architectures](#architectures) section below.

---

## Architectures

The active architecture is set by `ARCHITECTURE` in `config.py`. Each one supports independent teacher/student configurations.

### 1. Inception (`models/inception.py`)

Stacked Inception blocks with optional residual connections and a bottleneck layer.

```python
INCEPTION_TEACHER_DEPTH = 6     # teacher has 6 Inception blocks
INCEPTION_STUDENT_DEPTH = 4     # student has 4 (≈33% compression)
INCEPTION_NB_FILTERS    = 32
INCEPTION_BOTTLENECK_SIZE = 32
INCEPTION_KERNEL_SIZE   = 40
```

Compression dimension: **depth** (number of blocks).

### 2. FCN (`models/fcn.py`)

A Fully Convolutional Network with flexible depth *and* width.

```python
FCN_TEACHER_FILTERS = [128, 256, 128]    # 3 layers
FCN_STUDENT_FILTERS = [20, 40, 20]       # 3 layers, ~15% filters → heavy width compression
FCN_TEACHER_KERNEL_SIZES = [8, 5, 3]
FCN_STUDENT_KERNEL_SIZES = None          # falls back to defaults
```

Three compression strategies are supported simultaneously:

| Strategy | Example | Notes |
|---|---|---|
| Width only | `[20, 40, 20]` vs `[128, 256, 128]` | Same depth, fewer filters |
| Depth only | `[128, 256]` vs `[128, 256, 128]` | Fewer layers, full filters |
| Depth + width | `[64, 128]` vs `[128, 256, 128]` | Maximum compression |
| Deeper, narrower | `[64, 96, 128, 96, 64]` | More layers, varied filters |

The main script reports the compression ratio at startup.

### 3. ConvTran (`models/convtran.py`)

A hybrid Conv → Transformer architecture with relative positional encoding (eRPE / tAPE).

```python
TEACHER_NUM_HEADS = 8
STUDENT_NUM_HEADS = 6
```

Compression dimension: **number of attention heads**. Embedding size, feed-forward width, and dropout are fixed inside the model (`emb_size=24`, `dim_ff=256`).

---

## Knowledge Distillation

The `Distiller` class (`distiller.py`) wraps a frozen teacher and a trainable student:

```python
distiller = Distiller(student, teacher, alpha=0.5, temperature=10.0, device='cuda')
```

For each batch:

1. Teacher logits are computed under `torch.no_grad()`.
2. Student logits are computed normally.
3. `DistillationLoss` combines:
   - `CrossEntropyLoss(student_logits, labels)`
   - `KLDivLoss(log_softmax(student/T), softmax(teacher/T))`

Note that the KL term in `nn.KLDivLoss(reduction='batchmean')` is **not** multiplied by `T²`, which differs from the original Hinton et al. formulation. If you replicate that paper exactly, scale the distillation loss by `T²`.

---

## Output Structure

Results are written to `PATH_OUT/<ARCHITECTURE>/results/`:

```
Results/<ARCH>/results/
├── teacher/
│   └── UCRArchive_2018_itr_<i>/<Dataset>/
│       ├── best_model.pth
│       ├── last_model.pth
│       ├── history.csv
│       ├── df_metrics.csv         # test accuracy, precision, recall, duration
│       ├── df_best_model.csv      # epoch / loss of best checkpoint
│       └── DONE                   # marker file (skip if exists)
├── student_alone/
│   └── UCRArchive_2018_itr_<i>/<Dataset>/...
└── student_kd/
    └── alpha_<α>/temperature_<T>/UCRArchive_2018_itr_<i>/<Dataset>/...
```

The `DONE` marker enables **safe resumption** — re-running `main.py` skips completed experiments.

### Best teacher selection

`utils.get_best_teacher()` scans the teacher iterations for a given dataset and picks the checkpoint with the **lowest training loss** (`df_best_model.csv`). Set `BEST_TEACHER_ONLY = True` in `config.py` to enable this (default).

---

## Command-Line Arguments

```bash
python main.py [--datasets D1 D2 ...]
               [--classifiers C1 C2 ...]
               [--iterations N]
               [--architecture {inception,fcn,convtran}]
```

| Flag | Default | Description |
|---|---|---|
| `--datasets` | All in `UNIVARIATE_DATASET_NAMES_2018` | Subset of datasets to run |
| `--classifiers` | `CLASSIFIERS` from config | One or more of `teacher`, `student_kd`, `student_alone` |
| `--iterations` | From `ITERATIONS` dict | Override iteration count for all classifiers |
| `--architecture` | `ARCHITECTURE` from config | Override the backbone |

Examples:

```bash
# Train teachers on three datasets, three iterations each
python main.py --classifiers teacher --datasets ACSF1 Coffee GunPoint --iterations 3

# Run KD students with FCN architecture
python main.py --architecture fcn --classifiers student_kd

# Full pipeline on a single dataset
python main.py --datasets ACSF1 --classifiers teacher student_kd student_alone
```

---

## Extending the Framework

### Add a new architecture

1. Create `models/<your_model>.py` with a `forward(x) -> logits` model class.
2. In `trainers/trainer.py`, add a `_build_<your_arch>_model(...)` method that handles all three model types (`teacher`, `student_alone`, `student_kd`).
3. Wire it into `Trainer.__init__` and add a config branch in `main.py` for the architecture-specific parameters.

### Add a new KD loss

Subclass `DistillationLoss` in `distiller.py` and plug it into `Distiller`. For example, to add feature-map distillation, expose intermediate activations from your model and add an MSE term between teacher/student features.

### Sweep more hyperparameters

Extend `ALPHA_LIST` and `TEMPERATURE_LIST` in `config.py` — the main loop already does the cartesian product. Results are organized into separate folders per `(α, T)` for easy comparison.

---

## Notes and Gotchas

- **`val_loader` is `None`** in `Trainer.fit`, so the "validation" columns in `history.csv` are zeros. The best model is selected by **training loss**. Test set evaluation happens once at the end via `save_logs`.
- **One-hot vs. integer labels** are both handled — labels are converted to indices before cross-entropy.
- **Z-normalization** is applied per sample inside `data_utils.create_data_loaders`.
- **Reproducibility**: `torch.manual_seed(42)` is set only for the train DataLoader; for full determinism, also seed NumPy, Python's `random`, and set `torch.backends.cudnn.deterministic = True`.
- **InceptionTime input channels**: the first block assumes 1 input channel (univariate). For multivariate data, adjust `Inception_block.__init__`.

---

## License

Add your license here (e.g. MIT, Apache-2.0).

## Citation

If you use this codebase in academic work, please cite the original architectures:

- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015
- Ismail Fawaz et al., *InceptionTime: Finding AlexNet for Time Series Classification*, 2020
- Wang et al., *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline* (FCN), 2017
- Foumani et al., *Improving Position Encoding of Transformers for Multivariate Time Series Classification* (ConvTran), 2023
