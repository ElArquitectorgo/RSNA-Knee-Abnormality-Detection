from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd
import torch
from config import *

import math
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class MRIAugment:
    """
    volume: (S, K, H, W) float32 en [0, 1] (tensor nuevo, no un mmap).
    Geometría: un único remuestreo por plano (rotación+escala+traslación), compartido
    por todos los cortes del plano para mantener la coherencia 3D.
    Intensidad: independiente por slot (cada slot es una secuencia distinta).
    """
    HFLIP_PLANES = {"Axial", "Coronal"}
    REVERSE_SLICE_PLANES = {"Sagittal"}

    def __init__(
        self,
        plane_types=None,
        p_affine=0.8, max_rotation=8.0, scale_range=(0.92, 1.08), max_translate=0.05,
        p_flip=0.5,
        p_bias=0.3, bias_strength=0.15,
        p_gamma=0.5, gamma_range=(0.8, 1.25),
        p_contrast=0.5, contrast_range=(0.85, 1.15),
        p_blur=0.2, blur_sigma=(0.4, 1.0),
        p_noise=0.3, noise_std=(0.005, 0.02),
    ):
        self.plane_types = list(plane_types) if plane_types is not None else [s[0] for s in SLOTS]
        self.p_affine, self.max_rotation = p_affine, max_rotation
        self.scale_range, self.max_translate = scale_range, max_translate
        self.p_flip = p_flip
        self.p_bias, self.bias_strength = p_bias, bias_strength
        self.p_gamma, self.gamma_range = p_gamma, gamma_range
        self.p_contrast, self.contrast_range = p_contrast, contrast_range
        self.p_blur, self.blur_sigma = p_blur, blur_sigma
        self.p_noise, self.noise_std = p_noise, noise_std

        self.groups = {}
        for i, plane in enumerate(self.plane_types):
            self.groups.setdefault(plane, []).append(i)
        self.hflip_idxs = [i for i, p in enumerate(self.plane_types) if p in self.HFLIP_PLANES]
        self.reverse_idxs = [i for i, p in enumerate(self.plane_types) if p in self.REVERSE_SLICE_PLANES]

    @staticmethod
    def _affine(x, angle, scale, tx, ty):
        """x: (N, 1, H, W). Misma transformación para los N cortes."""
        N, _, H, W = x.shape
        t = math.radians(angle)
        c, s = math.cos(t) / scale, math.sin(t) / scale
        theta = torch.tensor([[[c, -s, tx], [s, c, ty]]], dtype=x.dtype)   # (1, 2, 3)
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False).expand(N, -1, -1, -1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _bias_field(self, H, W):
        low = torch.rand(1, 1, 3, 3) * 2 - 1
        field = F.interpolate(low, size=(H, W), mode="bicubic", align_corners=False)
        return 1 + self.bias_strength * field[0]                            # (1, H, W)

    def __call__(self, volume):
        S, K, H, W = volume.shape

        # 1) Geometría (por plano)
        for plane, idxs in self.groups.items():
            if random.random() < self.p_affine:
                angle = random.uniform(-self.max_rotation, self.max_rotation)
                scale = random.uniform(*self.scale_range)
                # coordenadas normalizadas van de -1 a 1, así que una fracción f del tamaño = 2f
                tx = random.uniform(-self.max_translate, self.max_translate) * 2
                ty = random.uniform(-self.max_translate, self.max_translate) * 2
                sub = volume[idxs].reshape(-1, 1, H, W)
                sub = self._affine(sub, angle, scale, tx, ty)
                volume[idxs] = sub.reshape(len(idxs), K, H, W)

        # 2) Flip L-R coherente: una sola decisión para todo el estudio
        if random.random() < self.p_flip:
            if self.hflip_idxs:
                volume[self.hflip_idxs] = torch.flip(volume[self.hflip_idxs], dims=[-1])
            if self.reverse_idxs:
                volume[self.reverse_idxs] = torch.flip(volume[self.reverse_idxs], dims=[1])

        # 3) Intensidad, borrosidad y ruido (por slot/secuencia)
        for i in range(S):
            x = volume[i]                                                    # (K, H, W)
            if random.random() < self.p_bias:
                x = x * self._bias_field(H, W)
            if random.random() < self.p_gamma:
                x = x.clamp(0, 1) ** random.uniform(*self.gamma_range)
            if random.random() < self.p_contrast:
                x = x * random.uniform(*self.contrast_range)
            if random.random() < self.p_blur:
                sigma = random.uniform(*self.blur_sigma)
                k = 2 * math.ceil(3 * sigma) + 1
                x = TF.gaussian_blur(x.unsqueeze(1), [k, k], [sigma, sigma]).squeeze(1)
            if random.random() < self.p_noise:                               # ruido Rician
                std = random.uniform(*self.noise_std)
                x = torch.sqrt((x + torch.randn_like(x) * std) ** 2 + (torch.randn_like(x) * std) ** 2)
            volume[i] = x

        return volume.clamp_(0.0, 1.0)

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
                 windows_per_slot=WINDOWS_PER_SLOT, window_size=WINDOW_SIZE, aug_p=1):
        self.uids = list(uids)
        self.index_df = index_df.set_index("StudyInstanceUID")
        self.preprocessed_dir = Path(preprocessed_dir)
        self.keep_idxs = keep_idxs
        self.augment = augment

        self.jitter = 1
        self.aug_p = aug_p
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
        if training and random.random() < self.aug_p:
            volume = self.augment(volume)
            mask = mask.clone()

        jitter = self.jitter if training else 0
        slot_windows, win_mask = [], []
        for s in range(volume.shape[0]):
            n_w = WINDOWS_BY_SLOT[s]
            slot_windows.append(series_volume_to_windows(volume[s], n_w, self.window_size, jitter))
            win_mask.append(mask[s].repeat(n_w))

        volume = torch.cat(slot_windows, dim=0)           # (T, ws, H, W), T = sum(WINDOWS_BY_SLOT)
        mask = torch.cat(win_mask, dim=0)                 # (T,)
        return volume, mask, y