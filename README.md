# RSNA Knee Abnormality Detection

Multi-label classification of knee MRI studies. Given a full study (several DICOM series in different planes), the model predicts the probability of 12 findings:

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

The evaluation metric is the **macro AUC** (mean of the per-finding AUC).

## Results

An ensemble of the models from the first 3 cross-validation folds reaches a **macro AUC of 0.92**. A single fold model typically scores around 0.90, so ensembling adds roughly +0.01 to +0.02.

## How it works

### 1. Labels

The competition only provides structured labels for a small number of studies. Every other study comes with a free-text radiology report. The training labels are produced by an LLM that reads each report and outputs 12 probabilities (soft labels instead of hard 0/1). A report that says "possible tear" becomes an intermediate value, and a finding that is not mentioned receives a neutral value (about 0.25).

### 2. Preprocessing (run once, `preprocess.py`)

`preprocess.py` converts each study into a fixed-size tensor and stores them in shards (`preprocessed/shard_XXXX.pt`) together with an `index.pkl`:

- Up to 6 **slots** are selected per study. Each slot is a `(plane, fat suppression, fluid sensitive)` combination.
- For each series, 6% of the slices are dropped at each end and the rest are subsampled to `K = 24` slices.
- Each slice is center-cropped to 140 mm (using the DICOM `PixelSpacing`), normalized with the 1st-99th percentiles and resized to 224x224 as `uint8`.
- If a study has no series for a slot, the slot is filled with zeros and its mask is set to 0.

### 3. Dataset (`dataset.py`)

For each study:

1. Load the `(slots, K, H, W)` volume from its shard.
2. During training, apply `MRIAugment` (per-plane affine transform, coherent left-right flip across planes, bias field, gamma, contrast, blur and Rician noise), plus window jitter and random dropout of entire slots.
3. Split each slot into **windows** of 3 consecutive slices, treated as the 3 channels of an RGB image (`WINDOWS_BY_SLOT` sets how many windows per slot).
4. Return `(windows, per-window mask, labels)`.

### 4. Model (`model.py`)

- **EfficientNet-B0** backbone pretrained on ImageNet, applied to every window independently.
- A linear layer maps each window to 12 logits.
- Logits are **averaged over all valid windows** (the mask ignores empty slots) to produce the study-level prediction.
- AdamW optimizer, with weight decay applied only to linear and convolutional weights (not to biases or normalization layers).

### 5. Training (`trainer.py`, `main.py`)

- Distributed training with **DDP** (one process per GPU, launched with `torchrun`).
- Mixed precision (fp16) with `GradScaler`, BCE-with-logits loss (supports `pos_weight`).
- Distributed validation: each GPU evaluates its shard, predictions are gathered with `gather_all`, and the AUC is computed on the full validation set.
- The best checkpoint by validation AUC is saved, with early stopping and configurable patience.
- 5-fold cross-validation stratified for multi-label data (`MultilabelStratifiedKFold`).
- Two-phase recipe per fold: (1) mostly frozen backbone with a higher LR, (2) everything unfrozen with a lower LR, starting from the best phase-1 checkpoint.
- For the final prediction, the probabilities of the fold models are averaged (ensemble).

## Repository structure

```
config.py       Hyperparameters, slots and paths
preprocess.py   Builds the preprocessed shards from the raw DICOM series
dataset.py      Dataset, windowing and data augmentation
model.py        Model and optimizer
trainer.py      Training and validation loop with DDP
utils.py        Metrics (AUC) and helpers
main.py         Entry point: folds and training phases
```

## Usage

Main requirements: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `iterative-stratification`, `pydicom`, `tqdm`.

```bash
pip install -r requirements.txt
```

The data is **not** included in the repository (DICOM files, reports, labels and shards are listed in `.gitignore`). Place the competition data in the working directory and generate the shards first:

```bash
python preprocess.py
```

Train with 4 GPUs:

```bash
torchrun --nproc_per_node=4 main.py
```

Using specific GPUs only (for example GPUs 2 and 3):

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 main.py
```

## Configuration

Almost everything is controlled from `config.py`:

| Parameter | Description |
|---|---|
| `RANDOM_STATE` | Random seed; it also defines the folds |
| `ALL_SLOTS`, `DROP_IDXS` | Available slots and which ones are dropped |
| `WINDOWS_BY_SLOT`, `WINDOW_SIZE` | Windows per slot and slices per window |
| `LR`, `WEIGHT_DECAY`, `BATCH_SIZE`, `EPOCHS` | Training settings (`BATCH_SIZE` is per GPU) |
| `PATIENCE` | Epochs without improvement before stopping |

## Notes

- Validation AUC varies by about ±0.01 just by changing the random seed, so comparisons between variants using a single fold and a single seed are not conclusive.
- The hardest findings on validation are `MCL`, `Lateral OA`, `PF OA` and `Contusion`.