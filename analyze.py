import textwrap
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

labels = pd.read_csv("llm_labels_v4_blend.csv")

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
        slot_windows, win_mask = [], []
        for s in range(volume.shape[0]):
            n_w = WINDOWS_BY_SLOT[s]
            slot_windows.append(series_volume_to_windows(volume[s], n_w, self.window_size, jitter))
            win_mask.append(mask[s].repeat(n_w))

        volume = torch.cat(slot_windows, dim=0)           # (T, ws, H, W), T = sum(WINDOWS_BY_SLOT)
        mask = torch.cat(win_mask, dim=0)                 # (T,)
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
TARGET = "MCL"                      # repite luego con "Lateral OA", "PF OA"...

index_df = pd.read_pickle(PREPROCESSED_DIR / "index.pkl")
splits = get_cv_splits(labels)
val_uids = labels.iloc[splits[fold][1]]["StudyInstanceUID"].values
val_ds = KneeStudyDataset(val_uids, index_df, PREPROCESSED_DIR, augment=None)

model = create_model().to(device)
state = torch.load(f"best_model_fold{fold}.pt", map_location=device, weights_only=True)
model.load_state_dict({k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()})

y_prob, y_true = predict(model, val_ds, drop_slot=None, device=device)

j = TARGETS.index(TARGET)

reports = pd.read_csv("train.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID")["Report"]

df = pd.DataFrame({
    "uid": val_uids,
    "label": y_true[:, j],      # viene de llm_labels_v4_blend.csv, vía los shards
    "pred": y_prob[:, j],
})
df["report"] = df["uid"].map(reports)
print("sin informe:", df["report"].isna().sum())

fp = df[df.label < 0.3].nlargest(10, "pred")     # modelo dice lesión, etiqueta dice no
fn = df[df.label > 0.7].nsmallest(10, "pred")    # etiqueta dice lesión, modelo dice no

def show(rows, title, reveal=False):
    print(f"\n{'='*25} {title} {'='*25}")
    for r in rows.itertuples():
        print(f"\n--- [{r.uid[-8:]}] ---")
        print(textwrap.fill(str(r.report), 120))
        if reveal:
            print(f">>> etiqueta={r.label:.2f}   pred={r.pred:.2f}")

show(fp, "FALSOS POSITIVOS (pred alta, etiqueta baja)", reveal=True)
show(fn, "FALSOS NEGATIVOS (pred baja, etiqueta alta)", reveal=True)

import matplotlib.pyplot as plt

def load_study_volume(uid):
    row = index_df.set_index("StudyInstanceUID").loc[uid]
    shard = torch.load(PREPROCESSED_DIR / f"shard_{int(row['shard']):04d}.pt",
                       map_location="cpu", weights_only=False, mmap=True)
    li = int(row["local_idx"])
    return shard["volumes"][li], shard["masks"][li]      # (6, K, H, W) uint8, (6,)

def slot_grid(uid, slot=1, ncols=6, tag=""):
    """slot = índice dentro de SLOTS (ya filtrado); lo convierto al de ALL_SLOTS."""
    vols, masks = load_study_volume(uid)
    s_all = KEEP_IDXS[slot]
    if not masks[s_all]:
        print(f"{uid[-8:]}: slot {slot} vacío"); return
    sl = vols[s_all].numpy()                              # (K, H, W)
    K = sl.shape[0]
    nrows = int(np.ceil(K / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    for i, ax in enumerate(np.ravel(axs)):
        ax.axis("off")
        if i < K:
            ax.imshow(sl[i], cmap="gray", vmin=0, vmax=255)
            ax.set_title(str(i), fontsize=8)
    plt.suptitle(f"{uid[-8:]}  slot {SLOTS[slot]}  {tag}")
    plt.tight_layout()
    plt.savefig(f"grid_slot{slot}_{uid[-8:]}.png", dpi=90)
    plt.close(fig)

for r in fn.head(5).itertuples():
    slot_grid(r.uid, slot=1, tag=f"FN label={r.label:.2f} pred={r.pred:.2f}")