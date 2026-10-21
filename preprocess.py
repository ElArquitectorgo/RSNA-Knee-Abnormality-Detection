from pathlib import Path
import os

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from pydicom.pixels import pixel_array
from tqdm.auto import tqdm

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

# See how test.py generates full_df in case you don't have the pkl 
full_df = pd.read_pickle("full_df.pkl")
labels = pd.read_csv("labels_merged.csv")

DATA_DIR = Path("")
TRAIN_DIR = DATA_DIR / "train_series"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

SLOTS = [
    ("Sagittal", 1, 1),
    ("Coronal",  1, 1),
    ("Axial",    1, 1),
    ("Sagittal", 0, 1),
    ("Coronal",  0, 0),
    #("Sagittal", 0, 0),
]

K = 24
OUT_SIZE = 224
CROP_MM = 140.0
SHARD_SIZE = 500

assert labels["StudyInstanceUID"].is_unique, "There are duplicated StudyInstanceUIDs."

print(f"Studies: {len(labels)}")
print(f"Series: {full_df['SeriesInstanceUID'].nunique()}")

# ---------- Fast lookups ----------
labels_by_uid = labels.set_index("StudyInstanceUID")[TARGETS]


# SeriesInstanceUID -> pixel spacing
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

    #slope = float(getattr(slices[0][1], "RescaleSlope", 1.0) or 1.0)
    #intercept = float(getattr(slices[0][1], "RescaleIntercept", 0.0) or 0.0)
    monochrome1 = (
        str(getattr(slices[0][1], "PhotometricInterpretation", "")).upper()
        == "MONOCHROME1"
    )

    arrays = []

    for path, _ in slices:
        try:
            arr = pixel_array(path).astype(np.float32, copy=False)
            #arr = arr * slope + intercept

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

    # Invalid shape -> zero image
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
    #lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)

    # Keep interpolation in PyTorch, but convert to uint8 immediately afterwards.
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(1) # (n_slices, 1, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(1) # (n_slices, out_size, out_size)

    out = (t * 255.0).round().clamp_(0, 255).to(torch.uint8)

    return out

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
            series_dir = TRAIN_DIR / study_uid / series_uid
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


def preprocess_study(study_uid):
    slots_uids = select_slot_series(study_uid)
    study_volume, mask = build_study_volume(study_uid, slots_uids)

    y = torch.from_numpy(
        labels_by_uid.loc[study_uid].to_numpy(dtype=np.float32, copy=True)
    )

    return study_uid, study_volume, mask, y

def preprocess_one(uid):
    return preprocess_study(uid)


def preprocess_dataset(study_uids, output_dir, shard_size=250, max_workers=2):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_studies = len(study_uids)

    all_index = []

    for shard_start in range(0, n_studies, shard_size):
        shard_idx = shard_start // shard_size
        shard_path = output_dir / f"shard_{shard_idx:04d}.pt"

        if shard_path.exists():
            print(f"Skipping existing {shard_path}")
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            for local_idx, uid in enumerate(shard["uids"]):
                all_index.append({
                    "StudyInstanceUID": uid,
                    "shard": shard_idx,
                    "local_idx": local_idx,
                })
            del shard
            continue

        shard_end = min(shard_start + shard_size, n_studies)
        shard_uids = study_uids[shard_start:shard_end]
        n_in_shard = len(shard_uids)

        print(
            f"\nProcessing studies {shard_start}:{shard_end} "
            f"({n_in_shard} studies)"
        )

        # Preallocate shard
        shard_volumes = torch.empty(
            (n_in_shard, len(SLOTS), K , OUT_SIZE, OUT_SIZE),
            dtype=torch.uint8,
        )

        shard_masks = torch.empty(
            (n_in_shard, len(SLOTS)),
            dtype=torch.bool,
        )

        shard_ys = torch.empty(
            (n_in_shard, len(TARGETS)),
            dtype=torch.float32,
        )

        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
            results = executor.map(preprocess_one, shard_uids)

            for i, result in enumerate(tqdm(results, total=n_in_shard, desc=f"Shard {shard_idx:04d}")):
                uid, volume, mask, y = result

                shard_volumes[i].copy_(volume)
                shard_masks[i].copy_(mask)
                shard_ys[i].copy_(y)

        torch.save(
            {
                "uids": shard_uids,
                "volumes": shard_volumes,
                "masks": shard_masks,
                "ys": shard_ys,
            },
            shard_path,
        )

        print(f"Saved: {shard_path}")

        for local_idx, uid in enumerate(shard_uids):
            all_index.append({
                "StudyInstanceUID": uid,
                "shard": shard_idx,
                "local_idx": local_idx,
            })


    index_df = pd.DataFrame(all_index)
    index_df.to_pickle(output_dir / "index.pkl")

    print("\nPreprocessing finished.")
    print(f"Studies indexed: {len(index_df)}")
    print(f"Index saved to: {output_dir / 'index.pkl'}")

    return index_df


if __name__ == "__main__":
    study_uids = labels["StudyInstanceUID"].tolist()

    preprocess_dataset(
        study_uids,
        PREPROCESSED_DIR,
        shard_size=SHARD_SIZE,
        max_workers=8,
    )
