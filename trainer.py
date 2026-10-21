import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from config import *
from utils import macro_auc, get_per_class_auc, gather_all

def seed_worker(worker_id):
    rank = int(os.environ.get("RANK", 0))
    s = (torch.initial_seed() + 1000 * rank) % 2**32
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)

class Trainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset, ckpt_path, resume_from=None, epochs=EPOCHS, patience=PATIENCE, amp=True):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.is_main = self.global_rank == 0
        self.log = print if self.is_main else (lambda *a, **k: None)
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.train_loader = self._make_dataloader(train_dataset, train=True)
        self.val_loader = self._make_dataloader(val_dataset, train=False)

        self.ckpt_path, self.epochs, self.patience, self.amp = ckpt_path, epochs, patience, amp

        self.raw_model = model.to(self.device)
        if resume_from:
            self._load_checkpoint(resume_from)

        self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=amp)
        self.criterion = nn.BCEWithLogitsLoss()

        net = torch.compile(self.raw_model, dynamic=False)
        self.model = DDP(net, device_ids=[self.local_rank])
    
    def _make_dataloader(self, dataset, train=False):
        sampler = DistributedSampler(dataset, shuffle=train, seed=RANDOM_STATE)
        return DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            sampler=sampler,
            num_workers=4, 
            pin_memory=True, 
            persistent_workers=True,
            drop_last=train,
            worker_init_fn=seed_worker,
        )
    
    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        checkpoint = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in checkpoint.items()}
        self.raw_model.load_state_dict(checkpoint)
        self.log("Checkpoint recovered:", path)
    
    def _save_checkpoint(self):
        if self.is_main:
            torch.save(self.raw_model.state_dict(), self.ckpt_path)
            print("Checkpoint saved:", self.ckpt_path)

    def _run_batch(self, imgs, mask, y, train=True):
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.model(imgs, mask)
        loss = self.criterion(logits.float(), y)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return logits.float().detach(), loss.detach()

    def _run_epoch(self, epoch, data_loader, train=True):
        self.model.train(train)
        if train:
            data_loader.sampler.set_epoch(epoch)

        loss_sum = torch.zeros(1, device=self.device)
        probs, targets = [], [] # used only for validation

        pbar = tqdm(data_loader, desc=f"{'Train' if train else 'Val'} - Epoch {epoch+1}/{EPOCHS}", disable=not self.is_main)
        for imgs, mask, y in pbar:
            imgs, mask, y = imgs.to(self.device), mask.to(self.device), y.to(self.device)
            logits, loss = self._run_batch(imgs, mask, y, train)
            loss_sum += loss * imgs.size(0)

            if not train:
                probs.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(y.cpu().numpy())

            pbar.set_postfix(loss=loss.item())
        
        dist.all_reduce(loss_sum);
        mean_loss = loss_sum.item() / (len(data_loader.sampler) * dist.get_world_size())
        if train:
            return mean_loss, None, None

        return mean_loss, gather_all(np.concatenate(probs)), gather_all(np.concatenate(targets))

    def train(self, epochs=EPOCHS):
        best_auc = 0.0
        epochs_no_improve = 0
        for epoch in range(epochs):
            tr_loss, _, _ = self._run_epoch(epoch, self.train_loader, train=True)
            val_loss, y_prob, y_true = self._run_epoch(epoch, self.val_loader, train=False)
            auc = macro_auc(y_true, y_prob)
            self.log(f"Epoch {epoch+1}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_macro_auc={auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                self._save_checkpoint()
                if self.is_main:
                    get_per_class_auc(y_prob, y_true)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            stop = torch.tensor(int(epochs_no_improve >= self.patience), device=self.device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                self.log(f"Early stopping in epoch {epoch+1}")
                break

        return best_auc