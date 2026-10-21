import os, random
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from config import TARGETS

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) 

def gather_all(arr):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, arr)
    return np.concatenate(out, axis=0)
    
def macro_auc(y_true, y_prob, threshold=0.5):
    aucs = []
    for j in range(y_true.shape[1]):
        y_true_bin = (y_true[:, j] >= threshold).astype(int)
        if len(np.unique(y_true_bin)) > 1:
            aucs.append(roc_auc_score(y_true_bin, y_prob[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")

def per_class_auc(y_true, y_prob, target_names, threshold=0.5):
    aucs = {}
    for j, name in enumerate(target_names):
        y_true_bin = (y_true[:, j] >= threshold).astype(int)
        if len(np.unique(y_true_bin)) > 1:
            aucs[name] = roc_auc_score(y_true_bin, y_prob[:, j])
        else:
            aucs[name] = float("nan")
    return aucs

def get_per_class_auc(y_prob, y_true):
    print("\nAUC per class:")
    class_aucs = per_class_auc(y_true, y_prob, TARGETS)
    for name, auc in sorted(class_aucs.items(), key=lambda kv: (np.isnan(kv[1]), kv[1])):
        status = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"  {name:30s} {status}")