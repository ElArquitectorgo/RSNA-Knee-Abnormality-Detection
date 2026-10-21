import os
import torch
import pandas as pd
from torch.distributed import init_process_group, destroy_process_group
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from trainer import Trainer
from model import create_model, create_optimizer
from dataset import KneeStudyDataset, MRIAugment
from utils import set_seed
from config import *

PHASES = [
    dict(name="p1", freeze=True,  lr=3e-4),
    dict(name="p2", freeze=True, lr=3e-5),
]

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")

def get_cv_splits(labels):
    uids = labels["StudyInstanceUID"].values
    y = labels[TARGETS].values

    mskf = MultilabelStratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return list(mskf.split(uids, y))

def run_fold(fold, labels, index_df, train_idx, val_idx):
    train_uids = labels.iloc[train_idx]["StudyInstanceUID"].values
    val_uids = labels.iloc[val_idx]["StudyInstanceUID"].values
    if int(os.environ["RANK"]) == 0:
        print(f"\nFold {fold + 1}/{N_SPLITS}  Train: {len(train_uids)}  Val: {len(val_uids)}")

    prev_ckpt = None

    for phase in PHASES:
        search = f"best_model_fold{fold}_{phase['name']}.pt"
        if os.path.exists(search):
            prev_ckpt = search

    for phase in PHASES:
        ckpt = f"best_model_fold{fold}_{phase['name']}.pt"
        if ckpt == prev_ckpt:
            continue

        set_seed(RANDOM_STATE)
        train_ds = KneeStudyDataset(train_uids, index_df, PREPROCESSED_DIR, augment=MRIAugment(), aug_p=0.5)
        val_ds = KneeStudyDataset(val_uids, index_df, PREPROCESSED_DIR, augment=None)
        model = create_model(freeze_backbone=phase["freeze"])
        optimizer = create_optimizer(model, phase["lr"])
        trainer = Trainer(model, optimizer, train_ds, val_ds, ckpt_path=ckpt, resume_from=prev_ckpt)
        auc = trainer.train()
        if trainer.is_main:
            print(f"[fold {fold} {phase['name']}] mejor AUC: {auc:.4f}")

        prev_ckpt = ckpt
        del model, optimizer, train_ds, val_ds, trainer
        torch.cuda.empty_cache()


def main():
    setup_ddp()

    labels = pd.read_csv("llm_labels_v4_blend.csv")
    index_df = pd.read_pickle(PREPROCESSED_DIR / "index.pkl")
    splits = get_cv_splits(labels)

    for fold, (train_idx, val_idx) in enumerate(splits):
        if not fold == 0:
            continue
        run_fold(fold, labels, index_df, train_idx, val_idx)

    destroy_process_group()

if __name__ == "__main__":
    main()