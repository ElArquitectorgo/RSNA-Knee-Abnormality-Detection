import torch
import torch.nn as nn
import torchvision
from config import *

class Model(nn.Module):
    def __init__(self, backbone, windows_per_slot=WINDOWS_PER_SLOT):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1000, 12)
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

def create_model(freeze_backbone=False, n_unfreeze=6):
    bb = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")

    if freeze_backbone:
        for p in bb.parameters():
            p.requires_grad = False
        for blk in bb.features[-n_unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True
    
    return Model(bb)


def create_optimizer(model, lr):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv2d)
    
    for model_name, m in model.named_modules():
        for param_name, p in m.named_parameters(recurse=False):
            name = f"{model_name}.{param_name}" if model_name else param_name
            if param_name.endswith('bias'):
                no_decay.add(name)
            elif param_name.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(name)
            else:
                no_decay.add(name)

    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    inter = decay & no_decay
    assert not inter, f"parameters {inter} made it into both sets!"

    groups = [
        {"params": [param_dict[n] for n in sorted(decay) if n in param_dict], "weight_decay": WEIGHT_DECAY},
        {"params": [param_dict[n] for n in sorted(no_decay) if n in param_dict], "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(groups, lr=lr)