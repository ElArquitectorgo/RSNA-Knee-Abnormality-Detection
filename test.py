from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pydicom
from pydicom.multival import MultiValue

import torch
import torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF

from torch.utils.data import Dataset, DataLoader
from pydicom.pixels import pixel_array
from tqdm.auto import tqdm
from transformers import AutoModel
import kagglehub

import os
import re

DATA_DIR = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")

TEST_DIR = DATA_DIR / "test_series"

test_series = pd.read_csv(DATA_DIR / "test_series.csv")

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']

SLOTS = [
    ("Sagittal", 1, 1),
    ("Coronal",  1, 1),
    ("Axial",    1, 1),
    ("Sagittal", 0, 1),
    ("Coronal",  0, 0),
]

WINDOW_SIZE = 3
WINDOWS_PER_SLOT = 18
OUT_SIZE = 224
CROP_MM = 140.0
K = 24

elements_list = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesDescription",
    "ScanningSequence",
    "ScanOptions",
    "SequenceName",
    "RepetitionTime",
    "EchoTime",
    "PixelSpacing",
]

tags_list = [
    (0x0020, 0x000D),  # StudyInstanceUID
    (0x0020, 0x000E),  # SeriesInstanceUID
    (0x0008, 0x103E),  # Series Description
    (0x0018, 0x0020),  # Scanning Sequence
    (0x0018, 0x0022),  # Scan Options
    (0x0018, 0x0024),  # Sequence Name
    (0x0018, 0x0080),  # Repetition Time
    (0x0018, 0x0081),  # Echo Time
    (0x0028, 0x0030),  # Pixel Spacing
]

def extract_series_metadata(row):
    dicom_path = TEST_DIR / row.StudyInstanceUID / row.SeriesInstanceUID
    try:
        first_file = next(dicom_path.glob("*.dcm"))
    except StopIteration:
        return {name: None for name in elements_list}

    ds = pydicom.dcmread(first_file, stop_before_pixels=True)

    record = {}
    for name, tag in zip(elements_list, tags_list):
        content = ds.get(tag)
        if content is not None:
            value = content.value
            if isinstance(value, MultiValue):
                value = "|".join(str(v) for v in value)
        else:
            value = None
        record[name] = value
    return record

rows = list(test_series.itertuples())

with ThreadPoolExecutor(max_workers=4) as executor:
    records = list(tqdm(executor.map(extract_series_metadata, rows), total=len(rows)))

metadata_df = pd.DataFrame(records)

def annotate(df):
    FATSAT_TOKENS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
    FATSAT_WORDS = re.compile(
        r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|water excit|"
        r"\btirm\b|\bfatsup\b")
    T1_WORDS = re.compile(r"\bt1\b|\bt1w\b")
    T2_WORDS = re.compile(r"\bt2\b|\bt2w\b")
    PD_WORDS = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

    description = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    description = description.str.lower().str.replace(r"[_\-.]", " ", regex=True)

    scan_options = df["ScanOptions"].fillna("").str.upper().str.split("|")
    options_say_fatsat = scan_options.apply(lambda tokens: any(t.strip() in FATSAT_TOKENS for t in tokens))

    df["fat_suppressed"] = description.str.contains(FATSAT_WORDS) | options_say_fatsat

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    is_gradient_echo = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")

    df["weighting"] = np.select(
        [
            description.str.contains(T1_WORDS) & ~description.str.contains(T2_WORDS)
            & ~description.str.contains(PD_WORDS),
            description.str.contains(T2_WORDS) & ~description.str.contains(PD_WORDS),
            description.str.contains(PD_WORDS),
            is_gradient_echo,
            tr < 800,
            te > 60,
            tr >= 800,
        ],
        ["T1", "T2", "PD", "GRE", "T1", "T2", "PD"],
        default="unknown",
    )
    df["fluid_sensitive"] = df["weighting"].isin(["T2", "PD"])
    df["pixel_spacing"] = pd.to_numeric(df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan), errors="coerce")
    return df

metadata_df = annotate(metadata_df)
planes = pd.read_csv(DATA_DIR / "test_series.csv")
planes = planes.drop(columns=["Fluid_Sensitive", "Fat_Suppression"])

full_df = planes.merge(metadata_df, how='inner', on=['SeriesInstanceUID', 'StudyInstanceUID'])

pixel_spacing_by_series = (
    full_df.drop_duplicates("SeriesInstanceUID")
    .set_index("SeriesInstanceUID")["pixel_spacing"]
    .to_dict()
)

slot_lookup = (
    full_df
    .drop_duplicates(
        ["StudyInstanceUID",
         "Anatomical_Plane",
         "fat_suppressed",
         "fluid_sensitive"]
    )
    .set_index(
        ["StudyInstanceUID",
         "Anatomical_Plane",
         "fat_suppressed",
         "fluid_sensitive"]
    )["SeriesInstanceUID"]
    .to_dict()
)

class Model(nn.Module):
    def __init__(self, backbone, windows_per_slot=WINDOWS_PER_SLOT):
        super().__init__()
        self.backbone = backbone
        self.fc1 = torch.nn.Linear(1000, 12)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        B, n_slots, windows_per_slot = imgs.shape[:3]
        imgs = imgs.reshape(B, n_slots * windows_per_slot, *imgs.shape[3:])  # (B, S, window_size, H, W)
        mask = mask.repeat_interleave(windows_per_slot, dim=1)              # (B, S)

        S = imgs.shape[1]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0) # (B*S, window_size, H, W)
        x = (x - self.mean) / self.std
        x = self.backbone(x) # (B*S, 1000)
        x = self.fc1(x) # (B*S, O)
        x = x.view(B, S, -1) # (B, S, O)

        m = mask.unsqueeze(-1).float() # (B, S, 1)
        x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6) # (B, 12)

        return x # (B, O)

def create_model():
    bb = torchvision.models.efficientnet_b0(weights=None)
    return Model(bb)

def select_slot_series(study_uid, slots=SLOTS):
    return [
        slot_lookup.get((study_uid, plane, fat_suppressed, fluid_sensitive))
        for plane, fat_suppressed, fluid_sensitive in slots
    ]


def order_slices(slices):
    """Return DICOM slices in acquisition order."""
    if not slices:
        return []

    ds0 = slices[0][1]

    if hasattr(ds0, "ImagePositionPatient") and hasattr(ds0, "ImageOrientationPatient"):
        row_dir = np.asarray(ds0.ImageOrientationPatient[:3], dtype=np.float32)
        col_dir = np.asarray(ds0.ImageOrientationPatient[3:6], dtype=np.float32)
        normal = np.cross(row_dir, col_dir)

        return sorted(
            slices,
            key=lambda x: np.dot(
                np.asarray(x[1].ImagePositionPatient, dtype=np.float32),
                normal,
            ),
            reverse=True,
        )

    if hasattr(ds0, "SliceLocation"):
        return sorted(slices, key=lambda x: float(x[1].SliceLocation))

    if hasattr(ds0, "InstanceNumber"):
        return sorted(slices, key=lambda x: int(x[1].InstanceNumber), reverse=True)

    raise ValueError("Slices are missing position/orientation tags.")


def get_slices(path):
    """Read only DICOM headers, then sort paths."""
    slices = []
    for slice_path in path.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(
                slice_path,
                stop_before_pixels=True,
                specific_tags=[
                    "ImagePositionPatient",
                    "ImageOrientationPatient",
                    "SliceLocation",
                    "InstanceNumber",
                    "RescaleSlope",
                    "RescaleIntercept",
                    "PhotometricInterpretation",
                ],
            )
            slices.append((slice_path, ds))
        except Exception as e:
            print(f"WARNING: cannot read header {slice_path}: {e}")

    return order_slices(slices)

def load_pixels(slices):
    """Decode slices one by one."""
    if not slices:
        return []
        
    monochrome1 = (
        str(getattr(slices[0][1], "PhotometricInterpretation", "")).upper()
        == "MONOCHROME1"
    )

    arrays = []

    for path, _ in slices:
        try:
            arr = pixel_array(path).astype(np.float32, copy=False)

            if monochrome1:
                arr = arr.max() - arr

            arrays.append(arr)

        except Exception:
            arrays.append(None)

    return arrays

def create_series_volume(series_uid, slices, size=OUT_SIZE):
    """Normalize, crop and resize a 3D series."""
    series_pixels = load_pixels(slices)

    valid_idx = [i for i, p in enumerate(series_pixels) if p is not None]
    if not valid_idx:
        print(f"WARNING: no slices could be decoded for series {series_uid}")
        return None

    # Replace broken slices using the nearest valid slice.
    for i, p in enumerate(series_pixels):
        if p is None:
            nearest = min(valid_idx, key=lambda j: abs(j - i))
            series_pixels[i] = series_pixels[nearest]

    shape = series_pixels[0].shape

    # Invalid shape -> zero image, keeping behavior close to the original code.
    for i, p in enumerate(series_pixels):
        if p.shape != shape:
            series_pixels[i] = np.zeros(shape, dtype=np.float32)

    vol = np.stack(series_pixels, axis=0).astype(np.float32, copy=False)

    px = pixel_spacing_by_series.get(series_uid)
    if px is not None and np.isfinite(px) and px > 0:
        want = int(round(CROP_MM / px))
        h, w = shape

        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    sub = vol[:, ::4, ::4]
    lo, hi = np.percentile(sub, [1, 99])
    vol = np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)

    # Keep interpolation in PyTorch, but convert to uint8 immediately afterwards.
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(1) # (n_slices, 1, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(1) # (n_slices, out_size, out_size)

    out = (t * 255.0).round().clamp_(0, 255).to(torch.uint8)

    # Release intermediate tensors before returning.
    del t, sub, vol, series_pixels

    return out

def series_volume_to_windows(series_volume, windows_per_slot, window_size=WINDOW_SIZE):
    n = series_volume.shape[0]
    max_start = max(n - window_size, 0)
    starts = np.linspace(0, max_start, windows_per_slot).round().astype(int)
    windows = []
    for start in starts:
        windows.append(series_volume[start:start + window_size])

    return torch.stack(windows) # (windows_per_slot, window_size, H, W)

def subsample_to_budget(slices, K):
    n = len(slices)
    if n <= K:
        return slices
    idx = np.linspace(0, n - 1, K).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [slices[i] for i in idx]


def build_study_volume(study_uid, slots_uids, K=K, out_size=OUT_SIZE):
    study_volume = torch.zeros((len(SLOTS), K, out_size, out_size), dtype=torch.uint8)
    mask = torch.zeros(len(SLOTS), dtype=torch.uint8)

    for i, series_uid in enumerate(slots_uids):
        if series_uid is None:
            continue
        try:
            series_dir = TEST_DIR / study_uid / series_uid
            slices = get_slices(series_dir)
            slices = slices[int(len(slices) * 0.06): int(len(slices) * 0.94)]
            slices = subsample_to_budget(slices, K)

            vol = create_series_volume(series_uid, slices, size=out_size)  # (K, H, W) uint8
            if vol is None:
                continue

            study_volume[i, :vol.shape[0]].copy_(vol)
            mask[i] = 1.0
        except Exception as e:
            print(f"WARNING: {study_uid} / {series_uid}: {e}")

    return study_volume, mask

def series_volume_to_windows(series_volume, windows_per_slot, window_size=WINDOW_SIZE):
    """series_volume: (K, H, W) uint8."""
    n_slices = series_volume.shape[0]

    step = max(1, (n_slices - window_size) // (windows_per_slot - 1)) if windows_per_slot > 1 else 1

    windows = []
    for i in range(windows_per_slot):
        start_idx = min(i * step, n_slices - window_size)
        windows.append(series_volume[start_idx:start_idx + window_size])

    return torch.stack(windows)  # (windows_per_slot, window_size, H, W)
    
class KneeStudyDataset(Dataset):
    def __init__(self, labels_df):
        self.labels_df = labels_df.set_index("StudyInstanceUID")
        self.study_uids = self.labels_df.index.tolist()

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        slots_uids = select_slot_series(uid) # n_slots
        study_volume, mask = build_study_volume(uid, slots_uids) # (n_slots, K, H, W)
        slot_windows = [
            series_volume_to_windows(study_volume[s], WINDOWS_PER_SLOT, WINDOW_SIZE)
            for s in range(study_volume.shape[0])
        ]
        study_volume = torch.stack(slot_windows)  # (n_slots, windows_per_slot, window_size, H, W)
        return uid, study_volume, mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_series = pd.read_csv(DATA_DIR / "test.csv")
test_ds = KneeStudyDataset(test_series)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

CKPT_PATHS = ["/kaggle/input/models/mega2814s/final-ensemble/pytorch/default/1/best_model_fold0_9016.pt", "/kaggle/input/models/mega2814s/final-ensemble/pytorch/default/1/best_model_fold1_9049.pt", "/kaggle/input/models/mega2814s/final-ensemble/pytorch/default/1/best_model_fold2_9001.pt"]

models = []
for path in CKPT_PATHS:
    m = create_model().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    m.load_state_dict(state)
    m.eval()
    models.append(m)

rows = []
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
    for uid, imgs, mask in test_dl:
        imgs, mask = imgs.to(DEVICE), mask.to(DEVICE)
        probs = torch.stack([
            torch.sigmoid(m(imgs, mask).float()) for m in models
        ]).mean(dim=0)[0].cpu().numpy()
        rows.append([uid[0], *probs])

"""CKPT_PATH = "/kaggle/input/models/mega2814s/9065/pytorch/default/1/best_model_fold1_9065.pt"

model = create_model().to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()

rows = []
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
    for uid, imgs, mask in test_dl:
        imgs, mask = imgs.to(DEVICE), mask.to(DEVICE)
        probs = torch.sigmoid(model(imgs, mask))[0].cpu().numpy()

        rows.append([uid[0], *probs])"""

submission = pd.DataFrame(rows, columns=["StudyInstanceUID", *TARGETS])
submission.to_csv("submission.csv", index=False)
