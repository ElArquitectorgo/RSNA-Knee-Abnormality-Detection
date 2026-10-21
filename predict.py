import pandas as pd
import numpy as np
from pathlib import Path
import torchvision
import torch.nn as nn, torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import *

def series_volume_to_windows(series_volume, windows_per_slot, window_size=WINDOW_SIZE, jitter=0):
    n = series_volume.shape[0]
    max_start = max(n - window_size, 0)
    starts = np.linspace(0, max_start, windows_per_slot).round().astype(int)
    windows = []
    for start in starts:
        if jitter:
            start = int(np.clip(start + random.randint(-jitter, jitter), 0, max_start))

        windows.append(series_volume[start:start + window_size])

    return torch.stack(windows) # (windows_per_slot, window_size, H, W)

class KneeStudyDataset(Dataset):
    def __init__(self, uids, index_df, preprocessed_dir, keep_idxs=KEEP_IDXS, augment=None,
                 windows_per_slot=WINDOWS_PER_SLOT, window_size=WINDOW_SIZE):
        self.uids = list(uids)
        self.index_df = index_df.set_index("StudyInstanceUID")
        self.preprocessed_dir = Path(preprocessed_dir)
        self.keep_idxs = keep_idxs
        self.augment = augment

        self.jitter=1
        self.p_slot_drop=0.1
        
        self.windows_per_slot = windows_per_slot
        self.window_size = window_size

        self._loaded_shard_idx = None
        self._loaded_shard = None

        labels_df = pd.read_csv("labels_merged.csv")
        lab = labels_df.set_index("StudyInstanceUID")[TARGETS]
        self.label_lookup = {u: lab.loc[u].to_numpy(np.float32) for u in self.uids if u in lab.index}

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        row = self.index_df.loc[uid]
        shard_idx = int(row["shard"])
        local_idx = int(row["local_idx"])

        if shard_idx != self._loaded_shard_idx:
            shard_path = self.preprocessed_dir / f"shard_{shard_idx:04d}.pt"
            self._loaded_shard = torch.load(str(shard_path), map_location="cpu", weights_only=False, mmap=True)
            self._loaded_shard_idx = shard_idx

        shard = self._loaded_shard
        volume = shard["volumes"][local_idx][self.keep_idxs]   # (n_slots, K, H, W) uint8
        volume = volume.float().div_(255.0)
        mask = shard["masks"][local_idx][self.keep_idxs]       # (n_slots,)
        #y = shard["ys"][local_idx]
        y = torch.from_numpy(self.label_lookup[uid])

        training = self.augment is not None
        if training:
            volume = self.augment(volume)
            mask = mask.clone()
            valid = mask.nonzero().flatten().tolist()
            if len(valid) > 1 and random.random() < self.p_slot_drop:
                mask[random.choice(valid)] = 0

        jitter = self.jitter if training else 0
        slot_windows = [
            series_volume_to_windows(volume[s], self.windows_per_slot, self.window_size, jitter)
            for s in range(volume.shape[0])
        ]
        volume = torch.stack(slot_windows)
        return volume, mask, y

class Model(nn.Module):
    def __init__(self, backbone, windows_per_slot=WINDOWS_PER_SLOT):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(1000, 12)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask):
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:]).float()
        x = (x - self.mean) / self.std
        x = self.backbone(x) # (B*S, 1000)
        x = self.fc1(self.drop(x)) # (B*S, O)
        x = x.view(B, T, -1) # (B, S, O)

        m = mask.unsqueeze(-1).float() # (B, S, 1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6) # (B, 12)

def create_model():
    bb = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")

    for prm in bb.parameters():
        prm.requires_grad = False

    features = bb.features  # nn.Sequential
    total = len(features)
    for i in range(total - UNFREEZE_LAST, total):
        for p in features[i].parameters():
            p.requires_grad = True
    
    return Model(bb)

def get_cv_splits(labels):
    uids = labels["StudyInstanceUID"].values
    y = labels[TARGETS].values

    mskf = MultilabelStratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return list(mskf.split(uids, y))

def per_class_auc(y_true, y_prob, target_names, threshold=0.5):
    aucs = {}
    for j, name in enumerate(target_names):
        y_true_bin = (y_true[:, j] >= threshold).astype(int)
        if len(np.unique(y_true_bin)) > 1:
            aucs[name] = roc_auc_score(y_true_bin, y_prob[:, j])
        else:
            aucs[name] = float("nan")
    return aucs

@torch.no_grad()
def predict(model, ds, drop_slot=None, bs=8, device="cuda:0"):
    model.eval()
    P, Y = [], []
    for imgs, mask, y in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4):
        mask = mask.clone()
        if drop_slot is not None:
            mask[:, drop_slot] = 0
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs.to(device), mask.to(device))
        P.append(torch.sigmoid(logits.float()).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)
    
device = "cuda:0"
fold = 0
full_df = pd.read_pickle("full_df.pkl")
labels = pd.read_csv("llm_labels_v4_blend.csv")
# 1) UIDs de validación del fold (mismos splits que en entrenamiento)
index_df = pd.read_pickle(PREPROCESSED_DIR / "index.pkl")
splits = get_cv_splits(labels)
val_uids = labels.iloc[splits[fold][1]]["StudyInstanceUID"].values
val_ds = KneeStudyDataset(val_uids, index_df, PREPROCESSED_DIR, augment=None)

# 2) Modelo con los pesos del fold
model = create_model().to(device)
state = torch.load(f"best_model_fold{fold}.pt", map_location=device, weights_only=True)
model.load_state_dict({k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()})
model.eval()

# 3) Base y ablación por slot
y_prob, y_true = predict(model, val_ds, drop_slot=None, device=device)
base = per_class_auc(y_true, y_prob, TARGETS)
print("Macro base:", np.nanmean(list(base.values())).round(4))

for s, (plane, fs, fl) in enumerate(SLOTS):
    p, _ = predict(model, val_ds, drop_slot=s, device=device)
    a = per_class_auc(y_true, p, TARGETS)
    print(f"\nSin slot {s} ({plane}, fs={fs}, fl={fl})  macro Δ = "
          f"{np.nanmean(list(a.values())) - np.nanmean(list(base.values())):+.4f}")
    for t in TARGETS:
        print(f"  {t:18s} {a[t] - base[t]:+.3f}")
