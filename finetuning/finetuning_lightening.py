import os
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from glob import glob
from PIL import Image
from pytorch_lightning.core import LightningModule
from torchmetrics.classification import MultilabelAUROC
import CARZero.builder as builder
import CARZero
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from nltk.tokenize import RegexpTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import Linear as LoRALinear
import re
from finetuning_dataset import build_t2i_mcq_batch

import random
from typing import List, Tuple, Optional

def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) ) where Dir(1) is uniform over K classes.
    alpha: (N, K), alpha > 0
    return: (N,) KL per sample
    """
    device = alpha.device
    N, K = alpha.shape
    beta = torch.ones((1, K), device=device, dtype=alpha.dtype).expand_as(alpha)  # (N, K)

    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)  # (N, 1)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)    # (N, 1) == K

    # ln B(alpha) = sum lgamma(alpha_i) - lgamma(sum alpha)
    lnB_alpha = torch.sum(torch.lgamma(alpha), dim=1, keepdim=True) - torch.lgamma(sum_alpha)
    lnB_beta  = torch.sum(torch.lgamma(beta),  dim=1, keepdim=True) - torch.lgamma(sum_beta)

    # KL = lnB(beta) - lnB(alpha) + sum (alpha_i - beta_i) * (digamma(alpha_i) - digamma(sum_alpha))
    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha)

    kl = (lnB_beta - lnB_alpha) + torch.sum(
        (alpha - beta) * (digamma_alpha - digamma_sum_alpha),
        dim=1,
        keepdim=True
    )  # (N, 1)

    return kl.squeeze(1)  # (N,)

class MCQEDLDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
        self.prompts = [f"There is {cls.replace('_', ' ')} but no {neg_cls.replace('_', ' ')}." for cls in self.class_names[:-1] for neg_cls in self.class_names[:-1] if cls != neg_cls] + self.prompts 
 
    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.cfg.peft.enabled :
            self.print("Setting up PEFT for the student model...")
            self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
            
    def set_peft(self):
        r = self.cfg.peft.r
        alpha = self.cfg.peft.alpha
        dropout = self.cfg.peft.dropout
        adaptor_name = self.cfg.peft.adaptor_name
        
        self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")

        # if adaptor_name == "lora":
        #     apply_lora(
        #         self.CARZero_model, 
        #         r=r, 
        #         alpha=alpha, 
        #         dropout=dropout, 
        #         merge_weights=False
        #     )
        #     self.print("LoRA adapters applied to the CARZero model.")
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.CARZero_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.CARZero_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.CARZero_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def i2t_forward(self, batch):
        i2t_cls, t2i_cls = self.CARZero_model.i2t_mcq_forward(batch, i2t_only=self.cfg.model.CARZero.single_path)
        
        targets = batch["answer_idx"].to(self.device)
        
        logits = i2t_cls
        
        N = logits.size(0)

        alpha = torch.exp(logits) + 1 # (N, T)
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        
        alpha_y = alpha[torch.arange(N, device=self.device), targets].unsqueeze(1)  # (N,1)
        loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()
        
        y = F.one_hot(targets, num_classes=logits.size(1)).to(alpha.dtype)  # (N, T)
        tilde_alpha = y + (1.0 - y) * alpha  # (N, T)
        
        loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
        
        epoch = getattr(self, "current_epoch", 0)
        lam = min(1.0, float(epoch) / 15.0)
        lam = torch.tensor(lam, device=self.device, dtype=alpha.dtype)

        loss = loss_match + lam * loss_kl
        acc = (probs.argmax(dim=1) == targets).float().mean()
        
        return i2t_cls, t2i_cls, loss, acc
    
    def t2i_forward(self, batch):
        batch = build_t2i_mcq_batch(
            batch,
            self.tokenizer,
            self.prompts,
            self.class_names,
            max_length=self.cfg.data.text.word_num,
            num_negatives=2,
            no_hyb=self.cfg.data.text.no_hyb
            )
        
        if len(batch['imgs'].shape) != 5 :
            self.print(f"Unexpected image batch shape: {batch['imgs'].shape}")
            return None, None, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        i2t_cls, t2i_cls = self.CARZero_model.t2i_mcq_forward(batch, t2i_only=self.cfg.model.CARZero.single_path)
        
        targets = batch["answer_idx"].to(self.device)
        
        logits = t2i_cls
        
        N = logits.size(0)

        alpha = torch.exp(logits) + 1 # (N, T)
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        
        alpha_y = alpha[torch.arange(N, device=self.device), targets].unsqueeze(1)  # (N,1)
        loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()
        
        y = F.one_hot(targets, num_classes=logits.size(1)).to(alpha.dtype)  # (N, T)
        tilde_alpha = y + (1.0 - y) * alpha  # (N, T)
        
        loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
        
        epoch = getattr(self, "current_epoch", 0)
        lam = min(1.0, float(epoch) / 15.0)
        lam = torch.tensor(lam, device=self.device, dtype=alpha.dtype)

        loss = loss_match + lam * loss_kl
        acc = (probs.argmax(dim=1) == targets).float().mean()
        
        return i2t_cls, t2i_cls, loss, acc

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        weight = self.cfg.train.loss_weight
        
        i2t_logits_i2t, t2i_logits_i2t, i2t_loss, i2t_acc = self.i2t_forward(batch)
        i2t_logits_t2i, t2i_logits_t2i, t2i_loss, t2i_acc = self.t2i_forward(batch)

        ce_loss = weight * i2t_loss + (1 - weight) * t2i_loss
        
        self.log_dict({f"{split}/loss": ce_loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc},
                  prog_bar=True, on_epoch=True)
                 
        return ce_loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_text(
            self.dm.train_dataset.pos_prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi)
        pos_logits = torch.tensor(pos_logits, device=self.device)
        alpha = torch.exp(pos_logits) + 1 
        S = torch.sum(alpha, dim=1, keepdim=True)
        pos_probs  = alpha / S

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_text(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi) # (N, 14)
        neg_logits = torch.tensor(neg_logits, device=self.device)
        alpha_neg = torch.exp(neg_logits) + 1 
        S_neg = torch.sum(alpha_neg, dim=1, keepdim=True)
        neg_probs  = alpha_neg / S_neg
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.loss_weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        
class MCQ2DQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi)
            self.freeze_module()
        # if self.cfg.peft.enabled :
        #     self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.CARZero_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.CARZero_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.CARZero_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def _forward_one_prompt(self, imgs, txt_ids, txt_mask, txt_type):
        """
        imgs      : (B, C, H, W)
        txt_ids   : (B, L)
        txt_mask  : (B, L)
        txt_type  : (B, L) or None
        반환      : img_cls (B, D), txt_cls (B, D)
        """
        forward_dict = {
            "imgs": imgs,
            "caption_ids": txt_ids,
            "attention_mask": txt_mask
        }
        if txt_type is not None:
            forward_dict["token_type_ids"] = txt_type
        *_, i2t_cls, t2i_cls = self.CARZero_model(forward_dict)   # (B,), (B,)
        return i2t_cls, t2i_cls

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        ans_idx = batch["answer_idx"].to(self.device)      # (B,)
        imgs     = batch["imgs"].to(self.device)           # (B,3,H,W)
        ids_all  = batch["caption_ids_all"].to(self.device)      # (B,4,L)
        mask_all = batch["attention_mask_all"].to(self.device)
        type_all = batch.get("token_type_ids_all", None)
        
        if type_all is not None:
            type_all = type_all.to(self.device)

        B, N, L = ids_all.shape            # N = 4

        i2t_list, t2i_list = [], []

        # ───────── 4-프롬프트 순차 처리 ──────────
        for j in range(N):
            txt_ids  = ids_all[:, j, :]        # (B,L)
            txt_mask = mask_all[:, j, :]
            txt_type = type_all[:, j, :] if type_all is not None else None

            i2t_cls, t2i_cls = self._forward_one_prompt(imgs, txt_ids, txt_mask, txt_type)

            i2t_list.append(i2t_cls)        # 정답 InfoNCE용
            t2i_list.append(t2i_cls)

        i2ts = torch.stack(i2t_list, 1)   # (B,4,D)
        t2is = torch.stack(t2i_list, 1)   # (B,

        *_, i2t_cls, t2i_cls = self.CARZero_model(
            batch
        )

        info_loss = self.CARZero_model.calc_loss(i2t_cls, t2i_cls)

        row_idx = torch.arange(B, device=self.device)
        i2t_logits = i2ts[row_idx, :, row_idx]
        ce_loss = _mcq_loss(
            i2t_logits, ans_idx, reduction="mean"
        )
        
        w = self.cfg.train.loss_weight
        loss = w* info_loss + (1 - w) * ce_loss
        
        self.log_dict({f"{split}/loss": loss,
                   f"{split}/info": info_loss,
                   f"{split}/ce": ce_loss},
                  prog_bar=True, on_epoch=True)
                 
        return loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text, multi=True).values
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text, multi=True).values
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc", (pos_mean_auroc + neg_mean_auroc)/2, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/pos_auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/pos_auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        pos_mean_auroc = torch.mean(class_auroc)
        neg_class_auroc = self.neg_auroc_metric.compute()
        neg_mean_auroc = torch.mean(neg_class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/mean_auroc : {((pos_mean_auroc + neg_mean_auroc)/2).item():.4f}")
            
            self.print(f" - test/pos_mean_auroc : {pos_mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

            self.print(f" - test/neg_mean_auroc : {neg_mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()

class PretrainDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
            self.freeze_module()
        # if self.cfg.peft.enabled :
        #     self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.CARZero_model.fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        bce_loss, mean_auroc, class_auroc = self.metrics(batch, "val")
        return {
            "val_loss": loss.detach(),
            "val_bce_loss": bce_loss.detach(),
            "mean_auroc": mean_auroc.detach(),
            "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss, mean_auroc, class_auroc = self.metrics(batch, "test")
        return {
            "test_loss": loss.detach(),
            "test_bce_loss": bce_loss.detach(),
            "mean_auroc": mean_auroc.detach(),
            "class_auroc": class_auroc.detach()
        }

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls = self.CARZero_model(batch)
        loss = self.CARZero_model.calc_loss(
            img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls
        )

        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        
        return loss
    
    def metrics(self, batch, split):
        processes_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        similarity = CARZero.dqn_shot_classification(
            self.CARZero_model,
            batch["imgs"].to(self.device),
            processes_text,).values
        similarity = torch.tensor(similarity).to(self.device)
        labels = batch["label"].to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        probs = torch.sigmoid(similarity)
        preds = (probs > 0.5).float()
        
        self.auroc_metric.update(probs, labels.int())  # 배치 단위로 누적만
        
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_bce_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{split}_mean_auroc",
            mean_auroc,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        metrics = {f"{split}_auroc_{cls}": class_auroc[i] for i, cls in enumerate(self.dm.train_dataset.class_names)}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss, mean_auroc, class_auroc
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val_loss", "val_bce_loss", "val_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val_auroc_")}
            for key in class_metrics:
                class_name = key.replace("val_auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()

class CenterLossDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)

        # ===== Center Loss 설정 (기본값 포함) =====
        self.cl_enabled: bool = cfg.train.center.enabled
        self.cl_weight: float = float(cfg.train.center.weight)
        self.cl_embedding: str = str(cfg.train.center.embedding)
        self.nf_idx: Optional[int] = 14  # setup에서 resolve

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
            self.freeze_module()
        if self.dm is None:
            self.dm = self.trainer.datamodule

    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.CARZero_model.fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        bce_loss, mean_auroc, class_auroc = self.metrics(batch, "val")
        return {
            "val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            "mean/auroc": mean_auroc.detach(),
            "class/auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss, mean_auroc, class_auroc = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
            "mean/auroc": mean_auroc.detach(),
            "class/auroc": class_auroc.detach()
        }

    # ===== 배치 단위 per-label positive center loss =====
    @torch.no_grad()
    def _select_embedding(self, img, txt, i2t, t2i):
        if self.cl_embedding == 'i2t' :
            return i2t
        elif self.cl_embedding == 't2i' :
            return t2i
        elif self.cl_embedding == "txt":
            return txt
        else:
            return img

    def compute_nf_center_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, D]
        labels:     [B, C] (multi-hot in {0,1})
        - 오직 No Finding 클래스(nf_idx) 양성(=1) 샘플에 대해서만 (x - μ)^2의 평균을 계산.
        - 양성 수가 <= 1이면 0 반환.
        """
        assert self.nf_idx is not None, "No Finding index must be resolved in setup()."
        mask = labels[:, self.nf_idx].bool()  # [B]
        cnt = mask.sum()
        if cnt <= 1:
            return embeddings.new_zeros(())
        emb_pos = embeddings[mask]                  # [N_pos, D]
        center = emb_pos.mean(dim=0, keepdim=True)  # [1, D]
        dist2 = (emb_pos - center).pow(2).sum(dim=1)  # [N_pos]
        return dist2.mean()  # scalar

    def shared_step(self, batch, split):
        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls, i2t_attn, t2i_attn, i2t_emb, t2i_emb = self.CARZero_model(batch, feat=True)

        i2t_emb = i2t_emb.mean(dim=1)
        t2i_emb = t2i_emb.mean(dim=0)

        # 기본 손실
        base_loss = self.CARZero_model.calc_loss(
            i2t_cls, t2i_cls
        )

        total_loss = base_loss
        nf_center_loss_val = torch.tensor(0.0, device=self.device)

        if self.cl_enabled:
            labels = batch["label"].to(self.device).float()  # [B, C]
            if self.cl_embedding == "both":
                nf_center_loss_i2t = self.compute_nf_center_loss(i2t_emb, labels)
                nf_center_loss_t2i = self.compute_nf_center_loss(t2i_emb, labels)
                nf_center_loss_val = (nf_center_loss_i2t + nf_center_loss_t2i) / 2
                total_loss = total_loss + self.cl_weight * nf_center_loss_val
            else:
                emb = self._select_embedding(img_emb_g, text_emb_g, i2t_emb, t2i_emb)
                if emb.dim() == 2 and emb.size(0) == labels.size(0):
                    nf_center_loss_val = self.compute_nf_center_loss(emb, labels)
                    total_loss = total_loss + self.cl_weight * nf_center_loss_val
                else:
                    self.print(f"[Warn] NF center loss skipped: embedding shape {tuple(emb.shape)} not [B, D]")

        # 로깅
        self.log(f"{split}/loss", total_loss, on_epoch=True, on_step=False, logger=True, prog_bar=True)
        if self.cl_enabled:
            self.log(f"{split}/nf_center_loss", nf_center_loss_val.detach(), on_epoch=True, on_step=False, logger=True)
            self.log(f"{split}/base_loss", base_loss.detach(), on_epoch=True, on_step=False, logger=True)
        return total_loss
    
    def metrics(self, batch, split):
        processes_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        similarity = CARZero.dqn_shot_classification(
            self.CARZero_model,
            batch["imgs"].to(self.device),
            processes_text,).values
        similarity = torch.tensor(similarity).to(self.device)
        labels = batch["label"].to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        probs = torch.sigmoid(similarity)
        
        self.auroc_metric.update(probs, labels.int())
        
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}/bce_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{split}/mean_auroc",
            mean_auroc,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        metrics = {f"{split}/auroc_{cls}": class_auroc[i] for i, cls in enumerate(self.dm.train_dataset.class_names)}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss, mean_auroc, class_auroc
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

        self.auroc_metric.reset()

class MCQDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.cfg.peft.enabled :
            self.print("Setting up PEFT for the student model...")
            self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
            
    def set_peft(self):
        r = self.cfg.peft.r
        alpha = self.cfg.peft.alpha
        dropout = self.cfg.peft.dropout
        adaptor_name = self.cfg.peft.adaptor_name
        
        self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")

        if adaptor_name == "lora":
            apply_lora(
                self.CARZero_model, 
                r=r, 
                alpha=alpha, 
                dropout=dropout, 
                merge_weights=False
            )
            self.print("LoRA adapters applied to the CARZero model.")
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.CARZero_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.CARZero_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.CARZero_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def _forward_one_prompt(self, imgs, txt_ids, txt_mask, txt_type):
        """
        imgs      : (B, C, H, W)
        txt_ids   : (B, L)
        txt_mask  : (B, L)
        txt_type  : (B, L) or None
        반환      : img_cls (B, D), txt_cls (B, D)
        """
        forward_dict = {
            "imgs": imgs,
            "caption_ids": txt_ids,
            "attention_mask": txt_mask
        }
        if txt_type is not None:
            forward_dict["token_type_ids"] = txt_type
        *_, i2t_cls, t2i_cls = self.CARZero_model(forward_dict)   # (B,), (B,)
        return i2t_cls, t2i_cls

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        ans_idx = batch["answer_idx"].to(self.device)      # (B,)
        imgs     = batch["imgs"].to(self.device)          # (B,3,H,W)
        ids_all  = batch["caption_ids_all"].to(self.device)      # (B,4,L)
        mask_all = batch["attention_mask_all"].to(self.device)
        type_all = batch.get("token_type_ids_all", None)
        
        if type_all is not None:
            type_all = type_all.to(self.device)

        B, N, L = ids_all.shape            # N = 4

        i2t_list, t2i_list = [], []

        # ───────── 4-프롬프트 순차 처리 ──────────
        for j in range(N):
            txt_ids  = ids_all[:, j, :]        # (B,L)
            txt_mask = mask_all[:, j, :]
            txt_type = type_all[:, j, :] if type_all is not None else None

            i2t_cls, t2i_cls = self._forward_one_prompt(imgs, txt_ids, txt_mask, txt_type)

            i2t_list.append(i2t_cls)        # 정답 InfoNCE용
            t2i_list.append(t2i_cls)

        i2ts = torch.stack(i2t_list, 1)   # (B,4,D)
        t2is = torch.stack(t2i_list, 1)   # (B,

        pos_img = i2ts[torch.arange(B), ans_idx]   # (B,D)
        pos_txt = t2is[torch.arange(B), ans_idx]   # (B,D)

        info_loss = self.CARZero_model.calc_loss(pos_img, pos_txt)
        
        row_idx = torch.arange(B, device=self.device)
        i2t_logits = i2ts[row_idx, :, row_idx]
        t2i_logits = t2is[row_idx, :, row_idx]
        logits = (i2t_logits + t2i_logits) / 2.0
        #logits = t2i_logits
        ce_loss = _mcq_loss(
            logits, ans_idx, reduction="mean"
        )
        
        w = self.cfg.train.loss_weight
        loss = w * info_loss + (1 - w) * ce_loss
        
        self.log_dict({f"{split}/loss": loss,
                   f"{split}/info": info_loss,
                   f"{split}/ce": ce_loss},
                  prog_bar=True, on_epoch=True)
                 
        return loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text, multi=True).values
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text, multi=True).values
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()

class MCQOnlyDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
        self.prompts = [f"There is {cls.replace('_', ' ')} but no {neg_cls.replace('_', ' ')}." for cls in self.class_names[:-1] for neg_cls in self.class_names[:-1] if cls != neg_cls] + self.prompts 
 
    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.cfg.peft.enabled :
            self.print("Setting up PEFT for the student model...")
            self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
            
    def set_peft(self):
        r = self.cfg.peft.r
        alpha = self.cfg.peft.alpha
        dropout = self.cfg.peft.dropout
        adaptor_name = self.cfg.peft.adaptor_name
        
        self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")

        if adaptor_name == "lora":
            apply_lora(
                self.CARZero_model, 
                r=r, 
                alpha=alpha, 
                dropout=dropout, 
                merge_weights=False
            )
            self.print("LoRA adapters applied to the CARZero model.")
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.CARZero_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.CARZero_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.CARZero_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def i2t_forward(self, batch):
        i2t_cls, t2i_cls = self.CARZero_model.i2t_mcq_forward(batch, i2t_only=self.cfg.model.CARZero.single_path)
        
        logits = (i2t_cls + t2i_cls)/2 if self.cfg.model.CARZero.single_path == False else i2t_cls
        
        targets = batch["answer_idx"].to(self.device)
        
        loss = F.cross_entropy(logits, targets, reduction="mean")
        acc = (logits.argmax(dim=1) == targets).float().mean()
        
        return i2t_cls, t2i_cls, loss, acc
    
    def t2i_forward(self, batch):
        batch = build_t2i_mcq_batch(
            batch,
            self.tokenizer,
            self.prompts,
            self.class_names,
            max_length=self.cfg.data.text.word_num,
            num_negatives=2,
            no_hyb=self.cfg.data.text.no_hyb
            )
        
        if len(batch['imgs'].shape) != 5 :
            self.print(f"Unexpected image batch shape: {batch['imgs'].shape}")
            return None, None, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        i2t_cls, t2i_cls = self.CARZero_model.t2i_mcq_forward(batch, t2i_only=self.cfg.model.CARZero.single_path)
        
        logits = (i2t_cls + t2i_cls)/2 if self.cfg.model.CARZero.single_path == False else t2i_cls
        
        acc = (logits.argmax(dim=1) == batch["answer_idx"].to(self.device)).float().mean()
        
        loss = F.cross_entropy(logits, batch["answer_idx"].to(self.device), reduction="mean")
        
        return i2t_cls, t2i_cls, loss, acc

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        weight = self.cfg.train.loss_weight
        
        i2t_logits_i2t, t2i_logits_i2t, i2t_loss, i2t_acc = self.i2t_forward(batch)
        i2t_logits_t2i, t2i_logits_t2i, t2i_loss, t2i_acc = self.t2i_forward(batch)

        ce_loss = weight * i2t_loss + (1 - weight) * t2i_loss
        
        self.log_dict({f"{split}/loss": ce_loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc},
                  prog_bar=True, on_epoch=True)
                 
        return ce_loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.pos_prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi)
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi) # (N, 14)
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.loss_weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()

class MCQEDLDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']
        
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        
        self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
        
        self.pos_prompts = [f"There is {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        self.neg_prompts = [f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
 
    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.cfg.peft.enabled :
            self.print("Setting up PEFT for the student model...")
            self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
            
    def set_peft(self):
        r = self.cfg.peft.r
        alpha = self.cfg.peft.alpha
        dropout = self.cfg.peft.dropout
        adaptor_name = self.cfg.peft.adaptor_name
        
        self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")

        # if adaptor_name == "lora":
        #     apply_lora(
        #         self.CARZero_model, 
        #         r=r, 
        #         alpha=alpha, 
        #         dropout=dropout, 
        #         merge_weights=False
        #     )
        #     self.print("LoRA adapters applied to the CARZero model.")
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.CARZero_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.CARZero_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.CARZero_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def i2t_forward(self, batch):
        i2t_logits, t2i_logits = self.CARZero_model.i2t_mcq_forward(batch, i2t_only=False) # (N, T) (N, T)
        
        targets = batch["answer_idx"].to(self.device)
        
        loss_ce = F.cross_entropy(i2t_logits[:, :-1], targets, reduction='mean')
        acc = (i2t_logits[:,:-1].argmax(dim=1) == targets).float().mean()
        
        i2t_target_logits = i2t_logits[torch.arange(i2t_logits.size(0)), targets] # (N,)
        i2t_counter_logits = i2t_logits[:, -1]
        t2i_target_logits = t2i_logits[torch.arange(t2i_logits.size(0)), targets] # (N,)
        t2i_counter_logits = t2i_logits[:, -1]
        
        i2t_beta_logits = torch.stack([i2t_target_logits, i2t_counter_logits], dim=1) # (N, 2)
        t2i_beta_logits = torch.stack([t2i_target_logits, t2i_counter_logits], dim=1) # (N, 2)
        
        beta_logits = (i2t_beta_logits/torch.exp(self.CARZero_model.i2t_tau) + t2i_beta_logits/torch.exp(self.CARZero_model.t2i_tau)) / 2 # (N, 2)
        
        N = i2t_logits.size(0)

        alpha = F.softplus(beta_logits) + 1 # (N, 2)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        alpha_y = alpha[:, 0:1]  # (N,1)
        loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()
        
        beta_targets = torch.zeros(N, dtype=torch.long, device=alpha.device) # (N,) 0: 정답, 1: 반대 문구
        y = F.one_hot(beta_targets, num_classes=2).to(alpha.dtype)  # (N, 2)
        tilde_alpha = y + (1.0 - y) * alpha  # (N, T)
        
        loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
        
        loss_edl = loss_match + self.cfg.train.edl_weight * loss_kl
        
        return i2t_logits, loss_ce, loss_edl, acc
    
    def t2i_forward(self, batch):
        batch = build_t2i_mcq_batch(
            batch,
            self.tokenizer,
            self.prompts,
            self.class_names,
            max_length=self.cfg.data.text.word_num,
            num_negatives=2,
            no_hyb=True
            )
        
        if len(batch['imgs'].shape) != 5 :
            self.print(f"Unexpected image batch shape: {batch['imgs'].shape}")
            return None, None, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        _, logits = self.CARZero_model.t2i_mcq_forward(batch, t2i_only=True)
        
        targets = batch["answer_idx"].to(self.device)
        
        acc = (logits.argmax(dim=1) == targets).float().mean()
        
        loss = F.cross_entropy(logits, targets, reduction="mean")
        
        return logits, loss, acc

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        weight = self.cfg.train.weight
        
        i2t_logits, i2t_loss, edl_loss, i2t_acc = self.i2t_forward(batch)
        t2i_logits, t2i_loss, t2i_acc = self.t2i_forward(batch)
        
        epoch = self.current_epoch + 1
        lam = min(1.0, float(epoch) / self.cfg.train.lam)
        lam = torch.tensor(lam, device=self.device, dtype=edl_loss.dtype)

        loss = (1-lam)*(weight * i2t_loss + (1 - weight) * t2i_loss)+(lam * edl_loss)
        
        self.log_dict({f"{split}/loss": loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/edl_loss": edl_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc},
                  prog_bar=True, on_epoch=True)
                
        return loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_text(
            self.pos_prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi, ts=True)
        pos_logits = torch.tensor(pos_logits, device=self.device)
        alpha_pos = F.softplus(pos_logits) + 1 

        # ---------- Negative-prompt similarity ----------  
        neg_text = self.CARZero_model.process_text(
            self.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi, ts=True) # (N, 14)
        neg_logits = torch.tensor(neg_logits, device=self.device)
        alpha_neg = F.softplus(neg_logits) + 1 
        
        S = alpha_pos + alpha_neg
        
        pos_probs  = alpha_pos / S                             # 질환 존재 확률
        neg_probs  = alpha_neg / S                             # 질환 부재 확률
        
        U = 2 / S # (N, 14)
        
        U_mean = U.mean(dim=0)                                 # 클래스별 평균 불확실성
        
        targets = labels[:,:-1].int()                             # 질환 존재 → 1
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1
        
        probs = pos_probs > neg_probs
        
        acc = (probs.int() == targets).float().mean()

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  targets)
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, targets.float())
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        
        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/acc", acc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
        self.log_dict({f"{split}/Uncertainty_{c}":     U_mean[i]
                          for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
                   
            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        
class MCQEDLLightModel(LightningModule):
    def __init__(self, cfg, CARZero_model=None):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = CARZero_model
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.failure_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']
        
        self.pos_prompts = [f"There is {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        self.neg_prompts = [f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        
        self.prompts = [*self.pos_prompts, *self.neg_prompts]
        
        self.toks = self.tokenizer(
            self.prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        self.cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in self.toks["input_ids"]],
            dtype=torch.long
        )

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.cfg.peft.enabled :
            self.print("Setting up PEFT for the student model...")
            self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
            
    def set_peft(self):
        r = self.cfg.peft.r
        alpha = self.cfg.peft.alpha
        dropout = self.cfg.peft.dropout
        adaptor_name = self.cfg.peft.adaptor_name
        
        self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.CARZero_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.CARZero_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.CARZero_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def generate_i2t_mcq(self, labels_batch: torch.Tensor, shuffle=True) -> Tuple[torch.Tensor, torch.Tensor]:
        B = labels_batch.shape[0]
        num_diseases = labels_batch.shape[1] - 1
        device = labels_batch.device
        
        disease_labels = labels_batch[:, :num_diseases] # (B, 14)
        
        pos_idx_range = torch.arange(num_diseases).to(device)
        neg_idx_range = torch.arange(num_diseases, 2 * num_diseases).to(device)

        # 2. 정답 선택 로직 (기존 로직 이식)
        # 각 샘플별로 긍정문 정답이 존재하는지 확인 (Any)
        has_pos_true = (disease_labels == 1).any(dim=1) # (B,)
        has_neg_true = (disease_labels == 0).any(dim=1)
        # 50% 확률로 긍정문을 정답으로 쓸지 결정
        use_pos_preference = torch.rand(B).to(device) < 0.5
        # 실제로 긍정문을 정답으로 선택할 샘플들
        select_pos = (has_pos_true & use_pos_preference) | (~has_neg_true)

        # 3. 정답 인덱스 추출 (행별로 다름)
        # 긍정 정답을 쓸 샘플은 1인 위치에서, 나머지는 0인 위치(부정 정답)에서 랜덤하게 하나 선택
        # 이를 위해 각 샘플의 정답 후보들에 가중치를 주어 multinomial로 뽑습니다.
        weights = torch.where(select_pos.unsqueeze(1), 
                            (disease_labels == 1).float(), 
                            (disease_labels == 0).float())
        
        # 각 행에서 가중치가 있는 곳 중 하나를 랜덤 추출 (정답의 질환 번호)
        selected_disease_idx = torch.multinomial(weights, 1).squeeze(1) # (B,)
        
        # 실제 프롬프트 인덱스로 변환
        # select_pos인 행은 긍정문(0~13), 아니면 부정문(14~27) 인덱스 선택
        answer_indices = torch.where(select_pos, 
                                    selected_disease_idx, 
                                    selected_disease_idx + num_diseases)

        # 4. 오답(Wrong) 선택 (2개)
        # 오답 후보: 정답과 반대되는 성격의 문장들
        # (label=1이면 부정문이 오답, label=0이면 긍정문이 오답)
        false_matrix = torch.where(disease_labels == 1, neg_idx_range, pos_idx_range)
        
        # 각 행에서 오답 후보 14개 중 2개를 랜덤 추출
        wrong_offsets = torch.multinomial(torch.ones((B, num_diseases)).to(device), 2, replacement=False)
        wrong_indices = false_matrix.gather(1, wrong_offsets)

        # 5. 최종 구성 및 셔플
        choices = torch.cat([answer_indices.unsqueeze(1), wrong_indices], dim=1)
        
        if shuffle:
            shuffled_idx = torch.argsort(torch.rand(B, 3).to(device), dim=1)
            choices = choices.gather(1, shuffled_idx)
            targets = (choices == answer_indices.unsqueeze(1)).nonzero()[:, 1]
        else:
            targets = torch.zeros(B, dtype=torch.long, device=device)

        return choices, targets
    
    def generate_t2i_mcq(self,labels_batch: torch.Tensor, shuffle=True):
        """
        labels_batch: (B, 15) - 배치 내 이미지들의 레이블
        반환값:
            valid_prompt_indices: (N,) - 유효한(정답이 존재하는) 프롬프트 번호들
            image_choices: (N, 3) - 각 유효 프롬프트별 선택된 이미지 인덱스 [Ans, W1, W2]
            targets: (N,) - 3개 중 정답 위치
        """
        B = labels_batch.shape[0]
        num_diseases = 14
        device = labels_batch.device
        
        # 1. 28개 프롬프트에 대한 전체 정답 지도 생성 (28, B)
        # 0~13: Positive (label 1), 14~27: Negative (label 0)
        disease_labels = labels_batch[:, :num_diseases].T  # (14, B)
        
        # row 0~13: 긍정문 일치 여부, row 14~27: 부정문 일치 여부
        is_correct_map = torch.cat([
            (disease_labels == 1),  # Positive Prompts
            (disease_labels == 0)   # Negative Prompts
        ], dim=0) # (28, B)

        # 2. 유효한 프롬프트 필터링
        # 정답(True)이 하나 이상 있고, 오답(False)이 두 개 이상 있는 프롬프트만 선택
        has_ans = is_correct_map.any(dim=1)
        has_wrongs = (~is_correct_map).sum(dim=1) >= 2
        valid_mask = has_ans & has_wrongs # (28,)
        
        valid_prompt_indices = torch.where(valid_mask)[0]
        num_valid = valid_prompt_indices.size(0)
        
        if num_valid == 0:
            return None, None, None

        # 유효한 프롬프트에 대한 맵만 추출
        filtered_map = is_correct_map[valid_prompt_indices] # (num_valid, B)

        # 3. 이미지 선택 (모든 이미지가 최대한 활용되도록 가중치 부여 가능)
        # 여기서는 각 프롬프트별로 독립적으로 샘플링하되, multinomial을 통해 무작위성 확보
        ans_weights = filtered_map.float()
        wrong_weights = (~filtered_map).float()

        # 정답 이미지 1개씩 추출 (num_valid, 1)
        ans_img_idx = torch.multinomial(ans_weights, 1)
        
        # 오답 이미지 2개씩 추출 (num_valid, 2)
        wrong_img_indices = torch.multinomial(wrong_weights, 2, replacement=False)

        # 4. 최종 구성 및 셔플
        image_choices = torch.cat([ans_img_idx, wrong_img_indices], dim=1) # (num_valid, 3)

        if shuffle:
            # (num_valid, 3) 크기의 셔플 인덱스 생성
            shuffled_idx = torch.argsort(torch.rand(num_valid, 3).to(device), dim=1)
            image_choices = image_choices.gather(1, shuffled_idx)
            # 정답이 이동한 위치 추적
            targets = (image_choices == ans_img_idx).nonzero()[:, 1]
        else:
            targets = torch.zeros(num_valid, dtype=torch.long, device=device)

        return valid_prompt_indices, image_choices, targets
    
    def image_forward(self, batch) :
        imgs = batch["imgs"].to(self.device)
        img_emb_l, img_emb_g = self.CARZero_model.image_encoder_forward(imgs)
        return img_emb_l, img_emb_g
    
    def text_forward(self) :
        text_emb_l, text_emb_g, sents = self.CARZero_model.text_encoder_forward(self.toks['input_ids'].to(self.device),
                                                                                self.toks['attention_mask'].to(self.device),
                                                                                self.toks['token_type_ids'].to(self.device),
                                                                                )
        return text_emb_l, text_emb_g, sents
    
    def fusion_forward(self, img_l, img_g, txt_l, txt_g):
        """
        img_l: (B, D, H, W) - 이미지 로컬 특징
        img_g: (B, D)       - 이미지 글로벌 특징
        txt_l: (T, D, L)    - 텍스트 로컬 특징 (L: 토큰 길이)
        txt_g: (T, D)       - 텍스트 글로벌 특징
        """
        B = img_l.shape[0]
        T = txt_g.shape[0]
        D = img_g.shape[1]
        
        # 1. Local feature 준비 (Flatten & Permute)
        # img_l: (B, D, H*W) -> (B, S_img, D)
        img_l_flat = img_l.view(B, D, -1).permute(0, 2, 1) 
        # txt_l: (T, D, L) -> (T, S_txt, D)
        txt_l_flat = txt_l.permute(0, 2, 1)

        # 2. B x T 조합을 위한 확장 (Expansion)
        # (B, 1, S_img, D) -> (B, T, S_img, D) -> (B*T, S_img, D)
        img_l_exp = img_l_flat.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, D)
        # (1, T, S_txt, D) -> (B, T, S_txt, D) -> (B*T, S_txt, D)
        txt_l_exp = txt_l_flat.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * T, -1, D)

        # 3. Global Query 준비 (B*T 개로 복사 및 차원 조정)
        # 이미지 쿼리: (B, 1, D) -> (B, T, D) -> (1, B*T, D)
        img_g_q = img_g.unsqueeze(1).expand(-1, T, -1).reshape(1, B * T, D)
        # 텍스트 쿼리: (1, T, D) -> (B, T, D) -> (1, B*T, D)
        txt_g_q = txt_g.unsqueeze(0).expand(B, -1, -1).reshape(1, B * T, D)

        # 4. Fusion Module을 통한 유사도 계산
        # I2T: 이미지를 쿼리로 텍스트의 로컬 토큰들을 참조
        i2t_logit = self.CARZero_model.i2t_fusion_module(
            txt_l_exp, img_g_q, inside_repeat=False
        ).squeeze(-1).squeeze(-1) # (B*T)
        
        # T2I: 텍스트를 쿼리로 이미지의 로컬 패치들을 참조
        t2i_logit = self.CARZero_model.t2i_fusion_module(
            img_l_exp, txt_g_q, inside_repeat=False
        ).squeeze(-1).squeeze(-1) # (B*T)

        # 5. (B, T) 유사도 행렬로 재구성
        i2t_matrix = i2t_logit.view(B, T)
        t2i_matrix = t2i_logit.view(B, T)

        return i2t_matrix, t2i_matrix
        
    def i2t_forward(self, i2t_cls, labels):
        i2t_choices, i2t_targets = self.generate_i2t_mcq(labels.to(self.device))
        
        i2t_logits = i2t_cls.gather(1, i2t_choices.to(self.device)) # (N, 3)
        i2t_ce_loss = F.cross_entropy(i2t_logits, i2t_targets.to(self.device), reduction='mean')
        i2t_acc = (i2t_logits.argmax(dim=1) == i2t_targets.to(self.device)).float().mean()
        
        return i2t_ce_loss, i2t_acc
    
    def t2i_forward(self, t2i_cls, labels):
        t2i_prompt_indices, t2i_image_choices, t2i_targets = self.generate_t2i_mcq(labels.to(self.device))
        
        t2i_logits = t2i_cls.T[t2i_prompt_indices].gather(1, t2i_image_choices.to(self.device)) # (T, 3)
        t2i_ce_loss = F.cross_entropy(t2i_logits, t2i_targets.to(self.device), reduction='mean')
        t2i_acc = (t2i_logits.argmax(dim=1) == t2i_targets.to(self.device)).float().mean()
        
        return t2i_ce_loss, t2i_acc
    
    def edl_forward(self, i2t_cls, t2i_cls, labels):
        """
        i2t_cls, t2i_cls: (B, T) - T=28 (0~13: Pos, 14~27: Neg)
        labels: (B, 15) - 14개 질환 + 1개 No Finding
        """
        B = i2t_cls.size(0)
        num_diseases = 14

        # 1. 타겟 질환(0~13) 결정
        # 각 샘플이 가진 질환들(label=1) 중 하나를 선택
        disease_labels = labels[:, :num_diseases] # (B, 14)
        nf_mask = labels[:, -1] == 1 # No Finding 샘플 마스크 (B,)

        # 질환이 있는 샘플은 있는 것 중 하나, NF 샘플은 전체(14개) 중 하나 무작위 선택
        # weights: 질환이 있으면 해당 위치 1, NF면 모든 위치 1
        selection_weights = disease_labels.float()
        selection_weights[nf_mask] = 1.0 
        
        # target_disease_idx: (B,) 각 이미지당 비교할 질환 번호 (0~13)
        target_disease_idx = torch.multinomial(selection_weights, 1).squeeze(1)

        # 2. 긍정 vs 부정 로짓 추출
        # i2t와 t2i 평균 점수 계산 (B, T)
        avg_logits = (i2t_cls / torch.exp(self.CARZero_model.i2t_tau) + 
                    t2i_cls / torch.exp(self.CARZero_model.t2i_tau)) / 2

        # 동일 질환의 긍정(0~13) 및 부정(14~27) 인덱스
        pos_prompts_idx = target_disease_idx
        neg_prompts_idx = target_disease_idx + num_diseases

        pos_scores = avg_logits[torch.arange(B), pos_prompts_idx] # (B,)
        neg_scores = avg_logits[torch.arange(B), neg_prompts_idx] # (B,)

        # 3. 정답(Ground Truth) 설정
        # 질환이 있는 샘플(NF가 아님)은 긍정이 정답(0), NF 샘플은 부정이 정답(1)
        # beta_logits: [Pos_Score, Neg_Score] (B, 2)
        beta_logits = torch.stack([pos_scores, neg_scores], dim=1)
        
        # targets: 0 이면 긍정 프롬프트가 참, 1 이면 부정 프롬프트가 참
        # NF 이미지가 아니면(질환 유) 0, NF 이미지면 1
        beta_targets = nf_mask.long() 

        # 4. EDL Loss 계산
        alpha = F.softplus(beta_logits) + 1.0 # (B, 2)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 정답 클래스에 해당하는 alpha 값 선택
        alpha_y = alpha.gather(1, beta_targets.unsqueeze(1)) # (B, 1)
        
        pred = torch.argmax(beta_logits, dim=1)
        correct = (pred == beta_targets).float()
        acc = correct.mean()
        
        # Loss Match: 정답에 대한 증거 극대화
        loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()
        
        # Loss KL: 불확실성 정규화
        y = F.one_hot(beta_targets, num_classes=2).to(alpha.dtype)
        tilde_alpha = y + (1.0 - y) * alpha
        loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
        
        loss_edl = loss_match + self.cfg.train.edl_weight * loss_kl
        return loss_edl, acc
        
    def shared_step(self, batch, split):
        img_l, img_g = self.image_forward(batch)
        txt_l, txt_g, sents = self.text_forward()
        
        i2t_cls, t2i_cls = self.fusion_forward(img_l, img_g, txt_l, txt_g)
        
        i2t_loss, i2t_acc = self.i2t_forward(i2t_cls, batch["label"])
        
        t2i_loss, t2i_acc = self.t2i_forward(t2i_cls, batch["label"])
        
        edl_loss, edl_acc = self.edl_forward(i2t_cls, t2i_cls, batch["label"])
        
        weight = self.cfg.train.weight
        
        epoch = self.current_epoch + 1
        lam = min(1.0, float(epoch) / self.cfg.train.lam)
        lam = torch.tensor(lam, device=self.device, dtype=edl_loss.dtype)

        loss = (1-lam)*(weight * i2t_loss + (1 - weight) * t2i_loss)+(lam * edl_loss)
        
        self.log_dict({f"{split}/loss": loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/edl_loss": edl_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc,
                       f"{split}/edl_acc": edl_acc},
                  prog_bar=True, on_epoch=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }
    
    def inference(self, batch, split="val") :
        with torch.no_grad():
            img_l, img_g = self.image_forward(batch)
            txt_l, txt_g, sents = self.text_forward()
    
            i2t_cls, t2i_cls = self.fusion_forward(img_l, img_g, txt_l, txt_g)
            
            avg_logits = (i2t_cls / torch.exp(self.CARZero_model.i2t_tau) + 
                t2i_cls / torch.exp(self.CARZero_model.t2i_tau)) / 2       
        return avg_logits
        
    def metrics(self, batch, split):
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)
        
        all_logits = self.inference(batch, split=split)
        
        pos_logits = all_logits[:,:14]
        neg_logits = all_logits[:,14:]
        alpha_pos = F.softplus(pos_logits) + 1
        alpha_neg = F.softplus(neg_logits) + 1
        
        S = alpha_pos + alpha_neg
        
        pos_probs  = alpha_pos / S                             # 질환 존재 확률
        neg_probs  = alpha_neg / S                             # 질환 부재 확률
        
        U = 2 / S # (N, 14)
        
        U_mean = U.mean(dim=0)                                 # 클래스별 평균 불확실성
        
        targets = labels[:,:-1].int()                             # 질환 존재 → 1
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1
        
        probs = pos_probs > neg_probs
        
        failure_case = (probs.int() != targets).int()
        self.failure_auroc_metric.update(U, failure_case)
            
        acc = (probs.int() == targets).float().mean()

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  targets)
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, targets.float())
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        
        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f"{split}/acc", acc, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f"{split}/U_mean", U_mean.mean(), prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)

        self.log_dict({f"{split}/Uncertainty_{c}":     U_mean[i]
                          for i, c in enumerate(self.class_names[:-1])}, sync_dist=True, on_epoch=True, on_step=False)

        return bce_loss
        
    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            self.print("==== Validation Epoch End ====")
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
            class_auroc = self.auroc_metric.compute()
            neg_class_auroc = self.neg_auroc_metric.compute()
            failure_class_auroc = self.failure_auroc_metric.compute()
            pos_mean_auroc = class_auroc.mean()
            neg_mean_auroc = neg_class_auroc.mean()
            mean_auroc = (pos_mean_auroc + neg_mean_auroc)/2
            failure_mean_auroc = failure_class_auroc.mean()
            
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
            self.print(f" - val/mean_auroc: {mean_auroc.item():.4f}")
            self.print(f" - val/pos_mean_auroc: {pos_mean_auroc.item():.4f}")
            self.print(f" - val/neg_mean_auroc: {neg_mean_auroc.item():.4f}")
            self.print(f" - val/FD_mean_auroc: {failure_mean_auroc.item():.4f}")
            
            self.log(f"val/mean_auroc", mean_auroc, sync_dist=True)
            self.log(f"val/pos_mean_auroc", pos_mean_auroc, sync_dist=True)
            self.log(f"val/neg_mean_auroc", neg_mean_auroc, sync_dist=True)
            self.log(f"val/FD_mean_auroc", failure_mean_auroc, sync_dist=True)
            
            self.log_dict({f"val/auroc_{c}":     class_auroc[i]
                           for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
            self.log_dict({f"val/neg_auroc_{c}": neg_class_auroc[i]
                           for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
            self.log_dict({f"val/FD_auroc_{c}": failure_class_auroc[i]
                           for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            for i, c in enumerate(self.class_names[:-1]):
                self.print(f"    {c:<22}: {class_auroc[i].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            for i, c in enumerate(self.class_names[:-1]):
                self.print(f"    {c:<22}: {neg_class_auroc[i].item():.4f}")
                
            self.print(f" - Failure Detection Mean AUROC")
            for i, c in enumerate(self.class_names[:-1]):
                self.print(f"    {c:<22}: {failure_class_auroc[i].item():.4f}")
                
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        self.failure_auroc_metric.reset()
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")
                
            self.print(f" - test/FD_mean_auroc : {self.failure_auroc_metric.compute().mean().item():.4f}")
            failure_class_auroc = self.failure_auroc_metric.compute()
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {failure_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        self.failure_auroc_metric.reset()
        
class MCQEDLLightModel2(LightningModule):
    def __init__(self, cfg, CARZero_model=None):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = CARZero_model
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']
        
        self.pos_prompts = [f"There is {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        self.neg_prompts = [f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        
        self.prompts = [*self.pos_prompts, *self.neg_prompts]
        
        self.toks = self.tokenizer(
            self.prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        self.automatic_optimization = False

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.print("CARZero model loaded")
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def configure_optimizers(self):
        mcq_params = []
        mcq_params += [p for p in self.CARZero_model.img_encoder.parameters() if p.requires_grad]
        mcq_params += [p for p in self.CARZero_model.text_encoder.parameters() if p.requires_grad]
        mcq_params += [p for p in self.CARZero_model.i2t_fusion_module.parameters() if p.requires_grad]
        mcq_params += [p for p in self.CARZero_model.t2i_fusion_module.parameters() if p.requires_grad]
        
        edl_params = []
        edl_params += [p for p in self.CARZero_model.i2t_fusion_module.parameters() if p.requires_grad]
        edl_params += [p for p in self.CARZero_model.t2i_fusion_module.parameters() if p.requires_grad]
        edl_params.append(self.CARZero_model.i2t_tau)
        edl_params.append(self.CARZero_model.t2i_tau)
        edl_params.append(self.CARZero_model.alpha)
        
        
        optimizer_mcq = torch.optim.Adam(mcq_params, lr=self.lr, weight_decay=self.cfg.train.optimizer.weight_decay, betas=(0.9, 0.98))
        optimizer_edl = torch.optim.Adam(edl_params, lr=self.lr*0.1, weight_decay=self.cfg.train.optimizer.weight_decay, betas=(0.9, 0.98))
        
        #scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return [optimizer_mcq, optimizer_edl]
    
    def generate_i2t_mcq(self, labels_batch: torch.Tensor, shuffle=True) -> Tuple[torch.Tensor, torch.Tensor]:
        B = labels_batch.shape[0]
        num_diseases = labels_batch.shape[1] - 1
        device = labels_batch.device
        
        disease_labels = labels_batch[:, :num_diseases] # (B, 14)
        
        pos_idx_range = torch.arange(num_diseases).to(device)
        neg_idx_range = torch.arange(num_diseases, 2 * num_diseases).to(device)

        # 2. 정답 선택 로직 (기존 로직 이식)
        # 각 샘플별로 긍정문 정답이 존재하는지 확인 (Any)
        has_pos_true = (disease_labels == 1).any(dim=1) # (B,)
        has_neg_true = (disease_labels == 0).any(dim=1)
        # 50% 확률로 긍정문을 정답으로 쓸지 결정
        use_pos_preference = torch.rand(B).to(device) < 0.5
        # 실제로 긍정문을 정답으로 선택할 샘플들
        select_pos = (has_pos_true & use_pos_preference) | (~has_neg_true)

        # 3. 정답 인덱스 추출 (행별로 다름)
        # 긍정 정답을 쓸 샘플은 1인 위치에서, 나머지는 0인 위치(부정 정답)에서 랜덤하게 하나 선택
        # 이를 위해 각 샘플의 정답 후보들에 가중치를 주어 multinomial로 뽑습니다.
        weights = torch.where(select_pos.unsqueeze(1), 
                            (disease_labels == 1).float(), 
                            (disease_labels == 0).float())
        
        # 각 행에서 가중치가 있는 곳 중 하나를 랜덤 추출 (정답의 질환 번호)
        selected_disease_idx = torch.multinomial(weights, 1).squeeze(1) # (B,)
        
        # 실제 프롬프트 인덱스로 변환
        # select_pos인 행은 긍정문(0~13), 아니면 부정문(14~27) 인덱스 선택
        answer_indices = torch.where(select_pos, 
                                    selected_disease_idx, 
                                    selected_disease_idx + num_diseases)

        # 4. 오답(Wrong) 선택 (2개)
        # 오답 후보: 정답과 반대되는 성격의 문장들
        # (label=1이면 부정문이 오답, label=0이면 긍정문이 오답)
        false_matrix = torch.where(disease_labels == 1, neg_idx_range, pos_idx_range)
        
        # 각 행에서 오답 후보 14개 중 2개를 랜덤 추출
        wrong_offsets = torch.multinomial(torch.ones((B, num_diseases)).to(device), 2, replacement=False)
        wrong_indices = false_matrix.gather(1, wrong_offsets)

        # 5. 최종 구성 및 셔플
        choices = torch.cat([answer_indices.unsqueeze(1), wrong_indices], dim=1)
        
        if shuffle:
            shuffled_idx = torch.argsort(torch.rand(B, 3).to(device), dim=1)
            choices = choices.gather(1, shuffled_idx)
            targets = (choices == answer_indices.unsqueeze(1)).nonzero()[:, 1]
        else:
            targets = torch.zeros(B, dtype=torch.long, device=device)

        return choices, targets
    
    def generate_t2i_mcq(self,labels_batch: torch.Tensor, shuffle=True):
        """
        labels_batch: (B, 15) - 배치 내 이미지들의 레이블
        반환값:
            valid_prompt_indices: (N,) - 유효한(정답이 존재하는) 프롬프트 번호들
            image_choices: (N, 3) - 각 유효 프롬프트별 선택된 이미지 인덱스 [Ans, W1, W2]
            targets: (N,) - 3개 중 정답 위치
        """
        B = labels_batch.shape[0]
        num_diseases = 14
        device = labels_batch.device
        
        # 1. 28개 프롬프트에 대한 전체 정답 지도 생성 (28, B)
        # 0~13: Positive (label 1), 14~27: Negative (label 0)
        disease_labels = labels_batch[:, :num_diseases].T  # (14, B)
        
        # row 0~13: 긍정문 일치 여부, row 14~27: 부정문 일치 여부
        is_correct_map = torch.cat([
            (disease_labels == 1),  # Positive Prompts
            (disease_labels == 0)   # Negative Prompts
        ], dim=0) # (28, B)

        # 2. 유효한 프롬프트 필터링
        # 정답(True)이 하나 이상 있고, 오답(False)이 두 개 이상 있는 프롬프트만 선택
        has_ans = is_correct_map.any(dim=1)
        has_wrongs = (~is_correct_map).sum(dim=1) >= 2
        valid_mask = has_ans & has_wrongs # (28,)
        
        valid_prompt_indices = torch.where(valid_mask)[0]
        num_valid = valid_prompt_indices.size(0)
        
        if num_valid == 0:
            return None, None, None

        # 유효한 프롬프트에 대한 맵만 추출
        filtered_map = is_correct_map[valid_prompt_indices] # (num_valid, B)

        # 3. 이미지 선택 (모든 이미지가 최대한 활용되도록 가중치 부여 가능)
        # 여기서는 각 프롬프트별로 독립적으로 샘플링하되, multinomial을 통해 무작위성 확보
        ans_weights = filtered_map.float()
        wrong_weights = (~filtered_map).float()

        # 정답 이미지 1개씩 추출 (num_valid, 1)
        ans_img_idx = torch.multinomial(ans_weights, 1)
        
        # 오답 이미지 2개씩 추출 (num_valid, 2)
        wrong_img_indices = torch.multinomial(wrong_weights, 2, replacement=False)

        # 4. 최종 구성 및 셔플
        image_choices = torch.cat([ans_img_idx, wrong_img_indices], dim=1) # (num_valid, 3)

        if shuffle:
            # (num_valid, 3) 크기의 셔플 인덱스 생성
            shuffled_idx = torch.argsort(torch.rand(num_valid, 3).to(device), dim=1)
            image_choices = image_choices.gather(1, shuffled_idx)
            # 정답이 이동한 위치 추적
            targets = (image_choices == ans_img_idx).nonzero()[:, 1]
        else:
            targets = torch.zeros(num_valid, dtype=torch.long, device=device)

        return valid_prompt_indices, image_choices, targets
    
    def image_forward(self, batch) :
        imgs = batch["imgs"].to(self.device)
        img_emb_l, img_emb_g = self.CARZero_model.image_encoder_forward(imgs)
        return img_emb_l, img_emb_g
    
    def text_forward(self) :
        text_emb_l, text_emb_g, sents = self.CARZero_model.text_encoder_forward(self.toks['input_ids'].to(self.device),
                                                                                self.toks['attention_mask'].to(self.device),
                                                                                self.toks['token_type_ids'].to(self.device),
                                                                                )
        return text_emb_l, text_emb_g, sents
    
    def fusion_forward(self, img_l, img_g, txt_l, txt_g):
        """
        img_l: (B, D, H, W) - 이미지 로컬 특징
        img_g: (B, D)       - 이미지 글로벌 특징
        txt_l: (T, D, L)    - 텍스트 로컬 특징 (L: 토큰 길이)
        txt_g: (T, D)       - 텍스트 글로벌 특징
        """
        B = img_l.shape[0]
        T = txt_g.shape[0]
        D = img_g.shape[1]
        
        # 1. Local feature 준비 (Flatten & Permute)
        # img_l: (B, D, H*W) -> (B, S_img, D)
        img_l_flat = img_l.view(B, D, -1).permute(0, 2, 1) 
        # txt_l: (T, D, L) -> (T, S_txt, D)
        txt_l_flat = txt_l.permute(0, 2, 1)

        # 2. B x T 조합을 위한 확장 (Expansion)
        # (B, 1, S_img, D) -> (B, T, S_img, D) -> (B*T, S_img, D)
        img_l_exp = img_l_flat.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, D)
        # (1, T, S_txt, D) -> (B, T, S_txt, D) -> (B*T, S_txt, D)
        txt_l_exp = txt_l_flat.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * T, -1, D)

        # 3. Global Query 준비 (B*T 개로 복사 및 차원 조정)
        # 이미지 쿼리: (B, 1, D) -> (B, T, D) -> (1, B*T, D)
        img_g_q = img_g.unsqueeze(1).expand(-1, T, -1).reshape(1, B * T, D)
        # 텍스트 쿼리: (1, T, D) -> (B, T, D) -> (1, B*T, D)
        txt_g_q = txt_g.unsqueeze(0).expand(B, -1, -1).reshape(1, B * T, D)

        # 4. Fusion Module을 통한 유사도 계산
        # I2T: 이미지를 쿼리로 텍스트의 로컬 토큰들을 참조
        i2t_logit = self.CARZero_model.i2t_fusion_module(
            txt_l_exp, img_g_q, inside_repeat=False
        ).squeeze(-1).squeeze(-1) # (B*T)
        
        # T2I: 텍스트를 쿼리로 이미지의 로컬 패치들을 참조
        t2i_logit = self.CARZero_model.t2i_fusion_module(
            img_l_exp, txt_g_q, inside_repeat=False
        ).squeeze(-1).squeeze(-1) # (B*T)

        # 5. (B, T) 유사도 행렬로 재구성
        i2t_matrix = i2t_logit.view(B, T)
        t2i_matrix = t2i_logit.view(B, T)

        return i2t_matrix, t2i_matrix
        
    def i2t_forward(self, i2t_cls, labels):
        i2t_choices, i2t_targets = self.generate_i2t_mcq(labels.to(self.device))
        
        i2t_logits = i2t_cls.gather(1, i2t_choices.to(self.device)) # (N, 3)
        i2t_ce_loss = F.cross_entropy(i2t_logits, i2t_targets.to(self.device), reduction='mean')
        i2t_acc = (i2t_logits.argmax(dim=1) == i2t_targets.to(self.device)).float().mean()
        
        return i2t_ce_loss, i2t_acc
    
    def t2i_forward(self, t2i_cls, labels):
        t2i_prompt_indices, t2i_image_choices, t2i_targets = self.generate_t2i_mcq(labels.to(self.device))
        
        t2i_logits = t2i_cls.T[t2i_prompt_indices].gather(1, t2i_image_choices.to(self.device)) # (T, 3)
        t2i_ce_loss = F.cross_entropy(t2i_logits, t2i_targets.to(self.device), reduction='mean')
        t2i_acc = (t2i_logits.argmax(dim=1) == t2i_targets.to(self.device)).float().mean()
        
        return t2i_ce_loss, t2i_acc
    
    def edl_forward(self, i2t_cls, t2i_cls, labels, split="train"):
        """
        i2t_cls, t2i_cls: (B, T) - T=28 (0~13: Pos, 14~27: Neg)
        labels: (B, 15) - 14개 질환 + 1개 No Finding
        """
        B = i2t_cls.size(0)
        num_diseases = 14

        # 1. 타겟 질환(0~13) 결정
        # 각 샘플이 가진 질환들(label=1) 중 하나를 선택
        disease_labels = labels[:, :num_diseases] # (B, 14)
        nf_mask = labels[:, -1] == 1 # No Finding 샘플 마스크 (B,)

        # 질환이 있는 샘플은 있는 것 중 하나, NF 샘플은 전체(14개) 중 하나 무작위 선택
        # weights: 질환이 있으면 해당 위치 1, NF면 모든 위치 1
        selection_weights = disease_labels.float()
        selection_weights[nf_mask] = 1.0 
        
        # target_disease_idx: (B,) 각 이미지당 비교할 질환 번호 (0~13)
        target_disease_idx = torch.multinomial(selection_weights, 1).squeeze(1)

        # 2. 긍정 vs 부정 로짓 추출
        # i2t와 t2i 평균 점수 계산 (B, T)
        i2t_cls = i2t_cls / torch.exp(self.CARZero_model.i2t_tau)
        t2i_cls = t2i_cls / torch.exp(self.CARZero_model.t2i_tau)
        avg_logits = self.CARZero_model.alpha * i2t_cls + (1 - self.CARZero_model.alpha) * t2i_cls

        # 동일 질환의 긍정(0~13) 및 부정(14~27) 인덱스
        pos_prompts_idx = target_disease_idx
        neg_prompts_idx = target_disease_idx + num_diseases

        pos_scores = avg_logits[torch.arange(B), pos_prompts_idx] # (B,)
        neg_scores = avg_logits[torch.arange(B), neg_prompts_idx] # (B,)

        # 3. 정답(Ground Truth) 설정
        # 질환이 있는 샘플(NF가 아님)은 긍정이 정답(0), NF 샘플은 부정이 정답(1)
        # beta_logits: [Pos_Score, Neg_Score] (B, 2)
        beta_logits = torch.stack([pos_scores, neg_scores], dim=1)
        
        beta_targets = nf_mask.long() 

        # 4. EDL Loss 계산
        alpha = F.softplus(beta_logits) + 1.0 # (B, 2)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 정답 클래스에 해당하는 alpha 값 선택
        alpha_y = alpha.gather(1, beta_targets.unsqueeze(1)) # (B, 1)
        
        pred = torch.argmax(beta_logits, dim=1)
        correct = (pred == beta_targets).float()
        acc = correct.mean()
        
        # Loss Match: 정답에 대한 증거 극대화
        loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()
        
        # Loss KL: 불확실성 정규화
        y = F.one_hot(beta_targets, num_classes=2).to(alpha.dtype)
        tilde_alpha = y + (1.0 - y) * alpha
        loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
        
        U = 2 / S
        
        self.log(f"{split}/edl_loss_match", loss_match, prog_bar=True)
        self.log(f"{split}/edl_loss_kl", loss_kl, prog_bar=True)
        self.log(f"{split}/U_mean", U.mean(), prog_bar=True)
        
        loss_edl = loss_match + self.cfg.train.edl_weight * loss_kl
        return loss_edl, acc
        
    def training_step(self, batch, split="train"):
        split = "train"
        opt_mcq, opt_edl = self.optimizers()
        
        opt_mcq.zero_grad()
        img_l, img_g = self.image_forward(batch)
        txt_l, txt_g, sents = self.text_forward()
        
        i2t_cls, t2i_cls = self.fusion_forward(img_l, img_g, txt_l, txt_g)
        
        i2t_loss, i2t_acc = self.i2t_forward(i2t_cls, batch["label"])
        
        t2i_loss, t2i_acc = self.t2i_forward(t2i_cls, batch["label"])
        
        weight = self.cfg.train.weight
        mcq_loss = weight * i2t_loss + (1 - weight) * t2i_loss
        self.manual_backward(mcq_loss)
        opt_mcq.step()
        
        opt_edl.zero_grad()
        
        img_l_det = img_l.detach()
        img_g_det = img_g.detach()
        txt_l_det = txt_l.detach()
        txt_g_det = txt_g.detach()
        
        i2t_cls_det, t2i_cls_det = self.fusion_forward(img_l_det, img_g_det, txt_l_det, txt_g_det)
        
        edl_loss, edl_acc = self.edl_forward(i2t_cls_det, t2i_cls_det, batch["label"], split=split)
        
        self.manual_backward(edl_loss)
        opt_edl.step()
        
        weight = self.cfg.train.weight

        loss = i2t_loss * weight + t2i_loss * (1 - weight) + edl_loss
        
        self.log_dict({f"{split}/loss": loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/edl_loss": edl_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc,
                       f"{split}/edl_acc": edl_acc},
                  prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/bce_loss": bce_loss.detach(),
        }
        
    def inference(self, batch):
        with torch.no_grad():
            img_l, img_g = self.image_forward(batch)
            txt_l, txt_g, sents = self.text_forward()
            i2t_cls, t2i_cls = self.fusion_forward(img_l, img_g, txt_l, txt_g)
            i2t_cls = i2t_cls / torch.exp(self.CARZero_model.i2t_tau)
            t2i_cls = t2i_cls / torch.exp(self.CARZero_model.t2i_tau)
            avg_logits = self.CARZero_model.alpha * i2t_cls + (1 - self.CARZero_model.alpha) * t2i_cls
            return avg_logits
        
    def metrics(self, batch, split):
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)
        
        all_logits = self.inference(batch)
        pos_logits = all_logits[:,:14]
        neg_logits = all_logits[:,14:]
        alpha_pos = F.softplus(pos_logits) + 1
        alpha_neg = F.softplus(neg_logits) + 1
        
        S = alpha_pos + alpha_neg
        
        pos_probs  = alpha_pos / S                             # 질환 존재 확률
        neg_probs  = alpha_neg / S                             # 질환 부재 확률
        
        U = 2 / S # (N, 14)
        
        U_mean = U.mean(dim=0)                                 # 클래스별 평균 불확실성
        
        targets = labels[:,:-1].int()                             # 질환 존재 → 1
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1
        
        probs = pos_probs > neg_probs
        
        acc = (probs.int() == targets).float().mean()

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  targets)
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, targets.float())
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        
        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc   = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/acc", acc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/U_mean", U_mean.mean(), prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
        self.log_dict({f"{split}/Uncertainty_{c}":     U_mean[i]
                          for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
                   
            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        
class MCQEDLLightModel3(LightningModule):
    def __init__(self, cfg, CARZero_model=None):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = CARZero_model
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']
        
        self.pos_prompts = [f"There is {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        self.neg_prompts = [f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        
        self.prompts = [*self.pos_prompts, *self.neg_prompts]
        
        self.toks = self.tokenizer(
            self.prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        self.automatic_optimization = False

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.print("CARZero model loaded")
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def configure_optimizers(self):
        edl_params = [p for p in self.CARZero_model.parameters() if p.requires_grad]
        
        optimizer_edl = torch.optim.Adam(edl_params, lr=self.lr, weight_decay=self.cfg.train.optimizer.weight_decay, betas=(0.9, 0.98))
        
        #scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer_edl
    
    def generate_i2t_mcq(self, labels_batch: torch.Tensor, shuffle=True) -> Tuple[torch.Tensor, torch.Tensor]:
        B = labels_batch.shape[0]
        num_diseases = labels_batch.shape[1] - 1
        device = labels_batch.device
        
        disease_labels = labels_batch[:, :num_diseases] # (B, 14)
        
        pos_idx_range = torch.arange(num_diseases).to(device)
        neg_idx_range = torch.arange(num_diseases, 2 * num_diseases).to(device)

        # 2. 정답 선택 로직 (기존 로직 이식)
        # 각 샘플별로 긍정문 정답이 존재하는지 확인 (Any)
        has_pos_true = (disease_labels == 1).any(dim=1) # (B,)
        has_neg_true = (disease_labels == 0).any(dim=1)
        # 50% 확률로 긍정문을 정답으로 쓸지 결정
        use_pos_preference = torch.rand(B).to(device) < 0.5
        # 실제로 긍정문을 정답으로 선택할 샘플들
        select_pos = (has_pos_true & use_pos_preference) | (~has_neg_true)

        # 3. 정답 인덱스 추출 (행별로 다름)
        # 긍정 정답을 쓸 샘플은 1인 위치에서, 나머지는 0인 위치(부정 정답)에서 랜덤하게 하나 선택
        # 이를 위해 각 샘플의 정답 후보들에 가중치를 주어 multinomial로 뽑습니다.
        weights = torch.where(select_pos.unsqueeze(1), 
                            (disease_labels == 1).float(), 
                            (disease_labels == 0).float())
        
        # 각 행에서 가중치가 있는 곳 중 하나를 랜덤 추출 (정답의 질환 번호)
        selected_disease_idx = torch.multinomial(weights, 1).squeeze(1) # (B,)
        
        # 실제 프롬프트 인덱스로 변환
        # select_pos인 행은 긍정문(0~13), 아니면 부정문(14~27) 인덱스 선택
        answer_indices = torch.where(select_pos, 
                                    selected_disease_idx, 
                                    selected_disease_idx + num_diseases)

        # 4. 오답(Wrong) 선택 (2개)
        # 오답 후보: 정답과 반대되는 성격의 문장들
        # (label=1이면 부정문이 오답, label=0이면 긍정문이 오답)
        false_matrix = torch.where(disease_labels == 1, neg_idx_range, pos_idx_range)
        
        # 각 행에서 오답 후보 14개 중 2개를 랜덤 추출
        wrong_offsets = torch.multinomial(torch.ones((B, num_diseases)).to(device), 2, replacement=False)
        wrong_indices = false_matrix.gather(1, wrong_offsets)

        # 5. 최종 구성 및 셔플
        choices = torch.cat([answer_indices.unsqueeze(1), wrong_indices], dim=1)
        
        if shuffle:
            shuffled_idx = torch.argsort(torch.rand(B, 3).to(device), dim=1)
            choices = choices.gather(1, shuffled_idx)
            targets = (choices == answer_indices.unsqueeze(1)).nonzero()[:, 1]
        else:
            targets = torch.zeros(B, dtype=torch.long, device=device)

        return choices, targets
    
    def generate_t2i_mcq(self,labels_batch: torch.Tensor, shuffle=True):
        """
        labels_batch: (B, 15) - 배치 내 이미지들의 레이블
        반환값:
            valid_prompt_indices: (N,) - 유효한(정답이 존재하는) 프롬프트 번호들
            image_choices: (N, 3) - 각 유효 프롬프트별 선택된 이미지 인덱스 [Ans, W1, W2]
            targets: (N,) - 3개 중 정답 위치
        """
        B = labels_batch.shape[0]
        num_diseases = 14
        device = labels_batch.device
        
        # 1. 28개 프롬프트에 대한 전체 정답 지도 생성 (28, B)
        # 0~13: Positive (label 1), 14~27: Negative (label 0)
        disease_labels = labels_batch[:, :num_diseases].T  # (14, B)
        
        # row 0~13: 긍정문 일치 여부, row 14~27: 부정문 일치 여부
        is_correct_map = torch.cat([
            (disease_labels == 1),  # Positive Prompts
            (disease_labels == 0)   # Negative Prompts
        ], dim=0) # (28, B)

        # 2. 유효한 프롬프트 필터링
        # 정답(True)이 하나 이상 있고, 오답(False)이 두 개 이상 있는 프롬프트만 선택
        has_ans = is_correct_map.any(dim=1)
        has_wrongs = (~is_correct_map).sum(dim=1) >= 2
        valid_mask = has_ans & has_wrongs # (28,)
        
        valid_prompt_indices = torch.where(valid_mask)[0]
        num_valid = valid_prompt_indices.size(0)
        
        if num_valid == 0:
            return None, None, None

        # 유효한 프롬프트에 대한 맵만 추출
        filtered_map = is_correct_map[valid_prompt_indices] # (num_valid, B)

        # 3. 이미지 선택 (모든 이미지가 최대한 활용되도록 가중치 부여 가능)
        # 여기서는 각 프롬프트별로 독립적으로 샘플링하되, multinomial을 통해 무작위성 확보
        ans_weights = filtered_map.float()
        wrong_weights = (~filtered_map).float()

        # 정답 이미지 1개씩 추출 (num_valid, 1)
        ans_img_idx = torch.multinomial(ans_weights, 1)
        
        # 오답 이미지 2개씩 추출 (num_valid, 2)
        wrong_img_indices = torch.multinomial(wrong_weights, 2, replacement=False)

        # 4. 최종 구성 및 셔플
        image_choices = torch.cat([ans_img_idx, wrong_img_indices], dim=1) # (num_valid, 3)

        if shuffle:
            # (num_valid, 3) 크기의 셔플 인덱스 생성
            shuffled_idx = torch.argsort(torch.rand(num_valid, 3).to(device), dim=1)
            image_choices = image_choices.gather(1, shuffled_idx)
            # 정답이 이동한 위치 추적
            targets = (image_choices == ans_img_idx).nonzero()[:, 1]
        else:
            targets = torch.zeros(num_valid, dtype=torch.long, device=device)

        return valid_prompt_indices, image_choices, targets
    
    def image_forward(self, batch) :
        imgs = batch["imgs"].to(self.device)
        img_emb_l, img_emb_g = self.CARZero_model.image_encoder_forward(imgs)
        return img_emb_l, img_emb_g
    
    def text_forward(self) :
        text_emb_l, text_emb_g, sents = self.CARZero_model.text_encoder_forward(self.toks['input_ids'].to(self.device),
                                                                                self.toks['attention_mask'].to(self.device),
                                                                                self.toks['token_type_ids'].to(self.device),
                                                                                )
        return text_emb_l, text_emb_g, sents
    
    def fusion_forward(self, img_l, img_g, txt_l, txt_g):
        """
        img_l: (B, D, H, W) - 이미지 로컬 특징
        img_g: (B, D)       - 이미지 글로벌 특징
        txt_l: (T, D, L)    - 텍스트 로컬 특징 (L: 토큰 길이)
        txt_g: (T, D)       - 텍스트 글로벌 특징
        """
        B = img_l.shape[0]
        T = txt_g.shape[0]
        D = img_g.shape[1]
        
        # 1. Local feature 준비 (Flatten & Permute)
        # img_l: (B, D, H*W) -> (B, S_img, D)
        img_l_flat = img_l.view(B, D, -1).permute(0, 2, 1) 
        # txt_l: (T, D, L) -> (T, S_txt, D)
        txt_l_flat = txt_l.permute(0, 2, 1)

        # 2. B x T 조합을 위한 확장 (Expansion)
        # (B, 1, S_img, D) -> (B, T, S_img, D) -> (B*T, S_img, D)
        img_l_exp = img_l_flat.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, D)
        # (1, T, S_txt, D) -> (B, T, S_txt, D) -> (B*T, S_txt, D)
        txt_l_exp = txt_l_flat.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * T, -1, D)

        # 3. Global Query 준비 (B*T 개로 복사 및 차원 조정)
        # 이미지 쿼리: (B, 1, D) -> (B, T, D) -> (1, B*T, D)
        img_g_q = img_g.unsqueeze(1).expand(-1, T, -1).reshape(1, B * T, D)
        # 텍스트 쿼리: (1, T, D) -> (B, T, D) -> (1, B*T, D)
        txt_g_q = txt_g.unsqueeze(0).expand(B, -1, -1).reshape(1, B * T, D)

        # 4. Fusion Module을 통한 유사도 계산
        # I2T: 이미지를 쿼리로 텍스트의 로컬 토큰들을 참조
        i2t_logit = self.CARZero_model.i2t_fusion_module(
            txt_l_exp, img_g_q, inside_repeat=False
        ).squeeze(-1).squeeze(-1) # (B*T)
        
        # T2I: 텍스트를 쿼리로 이미지의 로컬 패치들을 참조
        t2i_logit = self.CARZero_model.t2i_fusion_module(
            img_l_exp, txt_g_q, inside_repeat=False
        ).squeeze(-1).squeeze(-1) # (B*T)

        # 5. (B, T) 유사도 행렬로 재구성
        i2t_matrix = i2t_logit.view(B, T)
        t2i_matrix = t2i_logit.view(B, T)

        return i2t_matrix, t2i_matrix
        
    def i2t_forward(self, i2t_cls, labels):
        i2t_choices, i2t_targets = self.generate_i2t_mcq(labels.to(self.device))
        
        i2t_logits = i2t_cls.gather(1, i2t_choices.to(self.device)) # (N, 3)
        i2t_ce_loss = F.cross_entropy(i2t_logits, i2t_targets.to(self.device), reduction='mean')
        i2t_acc = (i2t_logits.argmax(dim=1) == i2t_targets.to(self.device)).float().mean()
        
        return i2t_ce_loss, i2t_acc
    
    def t2i_forward(self, t2i_cls, labels):
        t2i_prompt_indices, t2i_image_choices, t2i_targets = self.generate_t2i_mcq(labels.to(self.device))
        
        t2i_logits = t2i_cls.T[t2i_prompt_indices].gather(1, t2i_image_choices.to(self.device)) # (T, 3)
        t2i_ce_loss = F.cross_entropy(t2i_logits, t2i_targets.to(self.device), reduction='mean')
        t2i_acc = (t2i_logits.argmax(dim=1) == t2i_targets.to(self.device)).float().mean()
        
        return t2i_ce_loss, t2i_acc
    
    def edl_forward(self, i2t_cls, t2i_cls, labels):
        """
        i2t_cls, t2i_cls: (B, T) - T=28 (0~13: Pos, 14~27: Neg)
        labels: (B, 15) - 14개 질환 + 1개 No Finding
        """
        B = i2t_cls.size(0)
        num_diseases = 14

        # 1. 타겟 질환(0~13) 결정
        # 각 샘플이 가진 질환들(label=1) 중 하나를 선택
        disease_labels = labels[:, :num_diseases] # (B, 14)
        nf_mask = labels[:, -1] == 1 # No Finding 샘플 마스크 (B,)

        # 질환이 있는 샘플은 있는 것 중 하나, NF 샘플은 전체(14개) 중 하나 무작위 선택
        # weights: 질환이 있으면 해당 위치 1, NF면 모든 위치 1
        selection_weights = disease_labels.float()
        selection_weights[nf_mask] = 1.0 
        
        # target_disease_idx: (B,) 각 이미지당 비교할 질환 번호 (0~13)
        target_disease_idx = torch.multinomial(selection_weights, 1).squeeze(1)

        # 2. 긍정 vs 부정 로짓 추출
        # i2t와 t2i 평균 점수 계산 (B, T)
        i2t_cls = i2t_cls / torch.exp(self.CARZero_model.i2t_tau)
        t2i_cls = t2i_cls / torch.exp(self.CARZero_model.t2i_tau)
        avg_logits = self.CARZero_model.alpha * i2t_cls + (1 - self.CARZero_model.alpha) * t2i_cls

        # 동일 질환의 긍정(0~13) 및 부정(14~27) 인덱스
        pos_prompts_idx = target_disease_idx
        neg_prompts_idx = target_disease_idx + num_diseases

        pos_scores = avg_logits[torch.arange(B), pos_prompts_idx] # (B,)
        neg_scores = avg_logits[torch.arange(B), neg_prompts_idx] # (B,)

        # 3. 정답(Ground Truth) 설정
        # 질환이 있는 샘플(NF가 아님)은 긍정이 정답(0), NF 샘플은 부정이 정답(1)
        # beta_logits: [Pos_Score, Neg_Score] (B, 2)
        beta_logits = torch.stack([pos_scores, neg_scores], dim=1)
        
        # targets: 0 이면 긍정 프롬프트가 참, 1 이면 부정 프롬프트가 참
        # NF 이미지가 아니면(질환 유) 0, NF 이미지면 1
        beta_targets = 1-nf_mask.long() 

        # 4. EDL Loss 계산
        alpha = F.softplus(beta_logits) + 1.0 # (B, 2)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 정답 클래스에 해당하는 alpha 값 선택
        alpha_y = alpha.gather(1, beta_targets.unsqueeze(1)) # (B, 1)
        
        pred = torch.argmax(beta_logits, dim=1)
        correct = (pred == beta_targets).float()
        acc = correct.mean()
        
        # Loss Match: 정답에 대한 증거 극대화
        loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()
        
        # Loss KL: 불확실성 정규화
        y = F.one_hot(beta_targets, num_classes=2).to(alpha.dtype)
        tilde_alpha = y + (1.0 - y) * alpha
        loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
        
        loss_edl = loss_match + self.cfg.train.edl_weight * loss_kl
        return loss_edl, acc
        
    def training_step(self, batch, split="train"):
        split = "train"
        opt_edl = self.optimizers()
        
        opt_edl.zero_grad()
        img_l, img_g = self.image_forward(batch)
        txt_l, txt_g, sents = self.text_forward()
        
        i2t_cls, t2i_cls = self.fusion_forward(img_l, img_g, txt_l, txt_g)
        
        i2t_loss, i2t_acc = self.i2t_forward(i2t_cls, batch["label"])
        
        t2i_loss, t2i_acc = self.t2i_forward(t2i_cls, batch["label"])
        
        weight = self.cfg.train.weight
        mcq_loss = weight * i2t_loss + (1 - weight) * t2i_loss
        
        edl_loss, edl_acc = self.edl_forward(i2t_cls, t2i_cls, batch["label"])
        
        self.manual_backward(edl_loss)
        opt_edl.step()
        
        weight = self.cfg.train.weight

        loss = i2t_loss * weight + t2i_loss * (1 - weight) + edl_loss
        
        self.log_dict({f"{split}/loss": loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/edl_loss": edl_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc,
                       f"{split}/edl_acc": edl_acc},
                  prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/bce_loss": bce_loss.detach(),
        }
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)
        
        processed_text = self.CARZero_model.process_text(
            self.prompts, self.device)
        all_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, processed_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi, ts=True) # (N, 28)
        all_logits = torch.tensor(all_logits, device=self.device)
        pos_logits = all_logits[:,:14]
        neg_logits = all_logits[:,14:]
        alpha_pos = F.softplus(pos_logits) + 1
        alpha_neg = F.softplus(neg_logits) + 1
        
        S = alpha_pos + alpha_neg
        
        pos_probs  = alpha_pos / S                             # 질환 존재 확률
        neg_probs  = alpha_neg / S                             # 질환 부재 확률
        
        U = 2 / S # (N, 14)
        
        U_mean = U.mean(dim=0)                                 # 클래스별 평균 불확실성
        
        targets = labels[:,:-1].int()                             # 질환 존재 → 1
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1
        
        probs = pos_probs > neg_probs
        
        acc = (probs.int() == targets).float().mean()

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  targets)
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, targets.float())
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        
        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc   = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/acc", acc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
        self.log_dict({f"{split}/Uncertainty_{c}":     U_mean[i]
                          for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
                   
            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()

class MCQEDLLightModel4(LightningModule):
    def __init__(self, cfg, CARZero_model=None):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = CARZero_model
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']
        
        self.pos_prompts = [f"There is {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        self.neg_prompts = [f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        
        self.prompts = [*self.pos_prompts, *self.neg_prompts]
        
        self.toks = self.tokenizer(
            self.prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        self.automatic_optimization = False

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.print("CARZero model loaded")
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def configure_optimizers(self):
        mcq_params = []
        mcq_params += [p for p in self.CARZero_model.img_encoder.parameters() if p.requires_grad]
        mcq_params += [p for p in self.CARZero_model.text_encoder.parameters() if p.requires_grad]
        mcq_params += [p for p in self.CARZero_model.i2t_fusion_module.parameters() if p.requires_grad]
        mcq_params += [p for p in self.CARZero_model.t2i_fusion_module.parameters() if p.requires_grad]
        
        edl_params = []
        edl_params += [p for p in self.CARZero_model.edl_head.parameters() if p.requires_grad]
        
        optimizer_mcq = torch.optim.Adam(mcq_params, lr=self.lr, weight_decay=self.cfg.train.optimizer.weight_decay, betas=(0.9, 0.98))
        optimizer_edl = torch.optim.Adam(edl_params, lr=self.lr*0.1, weight_decay=self.cfg.train.optimizer.weight_decay, betas=(0.9, 0.98))

        return [optimizer_mcq, optimizer_edl]
    
    def generate_i2t_mcq(self, labels_batch: torch.Tensor, shuffle=True) -> Tuple[torch.Tensor, torch.Tensor]:
        B = labels_batch.shape[0]
        num_diseases = labels_batch.shape[1] - 1
        device = labels_batch.device
        
        disease_labels = labels_batch[:, :num_diseases] # (B, 14)
        
        pos_idx_range = torch.arange(num_diseases).to(device)
        neg_idx_range = torch.arange(num_diseases, 2 * num_diseases).to(device)

        # 2. 정답 선택 로직 (기존 로직 이식)
        # 각 샘플별로 긍정문 정답이 존재하는지 확인 (Any)
        has_pos_true = (disease_labels == 1).any(dim=1) # (B,)
        has_neg_true = (disease_labels == 0).any(dim=1)
        # 50% 확률로 긍정문을 정답으로 쓸지 결정
        use_pos_preference = torch.rand(B).to(device) < 0.5
        # 실제로 긍정문을 정답으로 선택할 샘플들
        select_pos = (has_pos_true & use_pos_preference) | (~has_neg_true)

        # 3. 정답 인덱스 추출 (행별로 다름)
        # 긍정 정답을 쓸 샘플은 1인 위치에서, 나머지는 0인 위치(부정 정답)에서 랜덤하게 하나 선택
        # 이를 위해 각 샘플의 정답 후보들에 가중치를 주어 multinomial로 뽑습니다.
        weights = torch.where(select_pos.unsqueeze(1), 
                            (disease_labels == 1).float(), 
                            (disease_labels == 0).float())
        
        # 각 행에서 가중치가 있는 곳 중 하나를 랜덤 추출 (정답의 질환 번호)
        selected_disease_idx = torch.multinomial(weights, 1).squeeze(1) # (B,)
        
        # 실제 프롬프트 인덱스로 변환
        # select_pos인 행은 긍정문(0~13), 아니면 부정문(14~27) 인덱스 선택
        answer_indices = torch.where(select_pos, 
                                    selected_disease_idx, 
                                    selected_disease_idx + num_diseases)

        # 4. 오답(Wrong) 선택 (2개)
        # 오답 후보: 정답과 반대되는 성격의 문장들
        # (label=1이면 부정문이 오답, label=0이면 긍정문이 오답)
        false_matrix = torch.where(disease_labels == 1, neg_idx_range, pos_idx_range)
        
        # 각 행에서 오답 후보 14개 중 2개를 랜덤 추출
        wrong_offsets = torch.multinomial(torch.ones((B, num_diseases)).to(device), 2, replacement=False)
        wrong_indices = false_matrix.gather(1, wrong_offsets)

        # 5. 최종 구성 및 셔플
        choices = torch.cat([answer_indices.unsqueeze(1), wrong_indices], dim=1)
        
        if shuffle:
            shuffled_idx = torch.argsort(torch.rand(B, 3).to(device), dim=1)
            choices = choices.gather(1, shuffled_idx)
            targets = (choices == answer_indices.unsqueeze(1)).nonzero()[:, 1]
        else:
            targets = torch.zeros(B, dtype=torch.long, device=device)

        return choices, targets
    
    def generate_t2i_mcq(self,labels_batch: torch.Tensor, shuffle=True):
        """
        labels_batch: (B, 15) - 배치 내 이미지들의 레이블
        반환값:
            valid_prompt_indices: (N,) - 유효한(정답이 존재하는) 프롬프트 번호들
            image_choices: (N, 3) - 각 유효 프롬프트별 선택된 이미지 인덱스 [Ans, W1, W2]
            targets: (N,) - 3개 중 정답 위치
        """
        B = labels_batch.shape[0]
        num_diseases = 14
        device = labels_batch.device
        
        # 1. 28개 프롬프트에 대한 전체 정답 지도 생성 (28, B)
        # 0~13: Positive (label 1), 14~27: Negative (label 0)
        disease_labels = labels_batch[:, :num_diseases].T  # (14, B)
        
        # row 0~13: 긍정문 일치 여부, row 14~27: 부정문 일치 여부
        is_correct_map = torch.cat([
            (disease_labels == 1),  # Positive Prompts
            (disease_labels == 0)   # Negative Prompts
        ], dim=0) # (28, B)

        # 2. 유효한 프롬프트 필터링
        # 정답(True)이 하나 이상 있고, 오답(False)이 두 개 이상 있는 프롬프트만 선택
        has_ans = is_correct_map.any(dim=1)
        has_wrongs = (~is_correct_map).sum(dim=1) >= 2
        valid_mask = has_ans & has_wrongs # (28,)
        
        valid_prompt_indices = torch.where(valid_mask)[0]
        num_valid = valid_prompt_indices.size(0)
        
        if num_valid == 0:
            return None, None, None

        # 유효한 프롬프트에 대한 맵만 추출
        filtered_map = is_correct_map[valid_prompt_indices] # (num_valid, B)

        # 3. 이미지 선택 (모든 이미지가 최대한 활용되도록 가중치 부여 가능)
        # 여기서는 각 프롬프트별로 독립적으로 샘플링하되, multinomial을 통해 무작위성 확보
        ans_weights = filtered_map.float()
        wrong_weights = (~filtered_map).float()

        # 정답 이미지 1개씩 추출 (num_valid, 1)
        ans_img_idx = torch.multinomial(ans_weights, 1)
        
        # 오답 이미지 2개씩 추출 (num_valid, 2)
        wrong_img_indices = torch.multinomial(wrong_weights, 2, replacement=False)

        # 4. 최종 구성 및 셔플
        image_choices = torch.cat([ans_img_idx, wrong_img_indices], dim=1) # (num_valid, 3)

        if shuffle:
            # (num_valid, 3) 크기의 셔플 인덱스 생성
            shuffled_idx = torch.argsort(torch.rand(num_valid, 3).to(device), dim=1)
            image_choices = image_choices.gather(1, shuffled_idx)
            # 정답이 이동한 위치 추적
            targets = (image_choices == ans_img_idx).nonzero()[:, 1]
        else:
            targets = torch.zeros(num_valid, dtype=torch.long, device=device)

        return valid_prompt_indices, image_choices, targets
    
    def image_forward(self, batch) :
        imgs = batch["imgs"].to(self.device)
        img_emb_l, img_emb_g = self.CARZero_model.image_encoder_forward(imgs)
        return img_emb_l, img_emb_g
    
    def text_forward(self) :
        text_emb_l, text_emb_g, sents = self.CARZero_model.text_encoder_forward(self.toks['input_ids'].to(self.device),
                                                                                self.toks['attention_mask'].to(self.device),
                                                                                self.toks['token_type_ids'].to(self.device),
                                                                                )
        return text_emb_l, text_emb_g, sents
    
    def fusion_forward(self, img_l, img_g, txt_l, txt_g):
        """
        img_l: (B, D, H, W) - 이미지 로컬 특징
        img_g: (B, D)       - 이미지 글로벌 특징
        txt_l: (T, D, L)    - 텍스트 로컬 특징 (L: 토큰 길이)
        txt_g: (T, D)       - 텍스트 글로벌 특징
        """
        B = img_l.shape[0]
        T = txt_g.shape[0]
        D = img_g.shape[1]
        
        # 1. Local feature 준비 (Flatten & Permute)
        # img_l: (B, D, H*W) -> (B, S_img, D)
        img_l_flat = img_l.view(B, D, -1).permute(0, 2, 1) 
        # txt_l: (T, D, L) -> (T, S_txt, D)
        txt_l_flat = txt_l.permute(0, 2, 1)

        # 2. B x T 조합을 위한 확장 (Expansion)
        # (B, 1, S_img, D) -> (B, T, S_img, D) -> (B*T, S_img, D)
        img_l_exp = img_l_flat.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, D)
        # (1, T, S_txt, D) -> (B, T, S_txt, D) -> (B*T, S_txt, D)
        txt_l_exp = txt_l_flat.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * T, -1, D)

        # 3. Global Query 준비 (B*T 개로 복사 및 차원 조정)
        # 이미지 쿼리: (B, 1, D) -> (B, T, D) -> (1, B*T, D)
        img_g_q = img_g.unsqueeze(1).expand(-1, T, -1).reshape(1, B * T, D)
        # 텍스트 쿼리: (1, T, D) -> (B, T, D) -> (1, B*T, D)
        txt_g_q = txt_g.unsqueeze(0).expand(B, -1, -1).reshape(1, B * T, D)

        # 4. Fusion Module을 통한 유사도 계산
        # I2T: 이미지를 쿼리로 텍스트의 로컬 토큰들을 참조
        i2t_logit, _, i2t_feat = self.CARZero_model.i2t_fusion_module(
            txt_l_exp, img_g_q, inside_repeat=False, return_feat=True
        )
        
        # T2I: 텍스트를 쿼리로 이미지의 로컬 패치들을 참조
        t2i_logit, _, t2i_feat = self.CARZero_model.t2i_fusion_module(
            img_l_exp, txt_g_q, inside_repeat=False, return_feat=True
        )

        # 5. (B, T) 유사도 행렬로 재구성
        i2t_matrix = i2t_logit.squeeze(-1).squeeze(-1).view(B, T)
        t2i_matrix = t2i_logit.squeeze(-1).squeeze(-1).view(B, T)
        
        i2t_feat = i2t_feat.view(B, T, D)
        t2i_feat = t2i_feat.view(B, T, D)
        
        return i2t_matrix, t2i_matrix, i2t_feat, t2i_feat
        
    def i2t_forward(self, i2t_cls, labels):
        i2t_choices, i2t_targets = self.generate_i2t_mcq(labels.to(self.device))
        
        i2t_logits = i2t_cls.gather(1, i2t_choices.to(self.device)) # (N, 3)
        i2t_ce_loss = F.cross_entropy(i2t_logits, i2t_targets.to(self.device), reduction='mean')
        i2t_acc = (i2t_logits.argmax(dim=1) == i2t_targets.to(self.device)).float().mean()
        
        return i2t_ce_loss, i2t_acc
    
    def t2i_forward(self, t2i_cls, labels):
        t2i_prompt_indices, t2i_image_choices, t2i_targets = self.generate_t2i_mcq(labels.to(self.device))
        
        t2i_logits = t2i_cls.T[t2i_prompt_indices].gather(1, t2i_image_choices.to(self.device)) # (T, 3)
        t2i_ce_loss = F.cross_entropy(t2i_logits, t2i_targets.to(self.device), reduction='mean')
        t2i_acc = (t2i_logits.argmax(dim=1) == t2i_targets.to(self.device)).float().mean()
        
        return t2i_ce_loss, t2i_acc
    
    def edl_forward(self, i2t_feats, t2i_feats, labels, split="train"):
        """
        i2t_cls, t2i_cls: (B, T) - T=28 (0~13: Pos, 14~27: Neg)
        i2t_feats, t2i_feats: (B, T, D)
        labels: (B, 15) - 14개 질환 + 1개 No Finding
        """
        B = i2t_feats.size(0)
        num_diseases = i2t_feats.size(1) // 2  # 28 프롬프트 중 절반이 질환 관련
           
        edl_matrix = self.CARZero_model.edl_head(torch.cat([i2t_feats, t2i_feats], dim=-1)).squeeze(-1) # (B, T)

        # 1. 타겟 질환(0~13) 결정
        # 각 샘플이 가진 질환들(label=1) 중 하나를 선택
        disease_labels = labels[:, :num_diseases] # (B, 14)
        nf_mask = labels[:, -1] == 1 # No Finding 샘플 마스크 (B,)

        # 질환이 있는 샘플은 있는 것 중 하나, NF 샘플은 전체(14개) 중 하나 무작위 선택
        # weights: 질환이 있으면 해당 위치 1, NF면 모든 위치 1
        selection_weights = disease_labels.float()
        selection_weights[nf_mask] = 1.0 
        
        # target_disease_idx: (B,) 각 이미지당 비교할 질환 번호 (0~13)
        target_disease_idx = torch.multinomial(selection_weights, 1).squeeze(1)

        # 2. 긍정 vs 부정 로짓 추출
        # i2t와 t2i 평균 점수 계산 (B, T)

        # 동일 질환의 긍정(0~13) 및 부정(14~27) 인덱스
        pos_prompts_idx = target_disease_idx
        neg_prompts_idx = target_disease_idx + num_diseases

        pos_scores = edl_matrix[torch.arange(B), pos_prompts_idx] # (B,)
        neg_scores = edl_matrix[torch.arange(B), neg_prompts_idx] # (B,)

        # 3. 정답(Ground Truth) 설정
        # 질환이 있는 샘플(NF가 아님)은 긍정이 정답(0), NF 샘플은 부정이 정답(1)
        # beta_logits: [Pos_Score, Neg_Score] (B, 2)
        beta_logits = torch.stack([pos_scores, neg_scores], dim=1)
        
        beta_targets = nf_mask.long() 

        # 4. EDL Loss 계산
        alpha = F.softplus(beta_logits) + 1.0 # (B, 2)
        S = torch.sum(alpha, dim=1, keepdim=True) # (B, 1)
        U = 2 / S # (B, 1) Uncertainty 계산
        
        # 정답 클래스에 해당하는 alpha 값 선택
        alpha_y = alpha.gather(1, beta_targets.unsqueeze(1)) # (B, 1)
        
        pred = torch.argmax(beta_logits, dim=1)                # (B,)
        correct = (pred == beta_targets)                       # (B,) bool
        wrong = (~correct).float()                             # (B,) float {0,1}
        acc = correct.float().mean()

        loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1)  # (B,)

        y_onehot = F.one_hot(beta_targets, num_classes=2).to(alpha.dtype)    # (B, 2)
        tilde_alpha = y_onehot + (1.0 - y_onehot) * alpha                    # (B, 2)

        kl_per_sample = dirichlet_kl_to_uniform(tilde_alpha)                 # (B,)
        loss_kl = wrong * kl_per_sample                                      # (B,)

        alpha_min = torch.min(alpha, dim=1).values          # (B,)
        alpha_max = torch.max(alpha, dim=1).values          # (B,)
        r = (alpha_min / (alpha_max + 1e-8)).detach()       # (B,)  gradient 차단
        logS = torch.log(S + 1e-8).squeeze(1)               # (B,)

        loss_s = r * logS                                   # (B,)
        
        self.log(f"{split}/match_loss", loss_match.mean(), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log(f"{split}/kl_loss", loss_kl.mean(), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log(f"{split}/s_loss", loss_s.mean(), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log(f"{split}/U_mean", U.mean(), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        
        loss_edl = loss_match + self.cfg.train.edl_weight * loss_kl + 0.1*loss_s
        loss_edl = loss_edl.mean()
        
        return loss_edl, acc
        
    def training_step(self, batch, batch_idx):
        split = "train"
        opt_mcq, opt_edl = self.optimizers()
        
        opt_mcq.zero_grad()
        img_l, img_g = self.image_forward(batch)
        txt_l, txt_g, sents = self.text_forward()
        
        i2t_cls, t2i_cls, i2t_feats, t2i_feats = self.fusion_forward(img_l, img_g, txt_l, txt_g)
        
        i2t_loss, i2t_acc = self.i2t_forward(i2t_cls, batch["label"])
        
        t2i_loss, t2i_acc = self.t2i_forward(t2i_cls, batch["label"])
        
        weight = self.cfg.train.weight
        mcq_loss = weight * i2t_loss + (1 - weight) * t2i_loss
        
        self.manual_backward(mcq_loss)
        opt_mcq.step()
        
        opt_edl.zero_grad()
        
        i2t_feats = i2t_feats.detach()  # EDL 계산에서는 MCQ 피쳐 고정
        t2i_feats = t2i_feats.detach()
        edl_loss, edl_acc = self.edl_forward(i2t_feats, t2i_feats, batch["label"])
        
        self.manual_backward(edl_loss)
        opt_edl.step()
        
        weight = self.cfg.train.weight

        loss = i2t_loss + t2i_loss + edl_loss
        
        self.log_dict({f"{split}/loss": loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/edl_loss": edl_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc,
                       f"{split}/edl_acc": edl_acc},
                  prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/bce_loss": bce_loss.detach(),
        }
        
    def inference(self, batch, split="val"):
        with torch.no_grad():
            img_l, img_g = self.image_forward(batch)
            txt_l, txt_g, sents = self.text_forward()
            
            i2t_cls, t2i_cls, i2t_feats, t2i_feats = self.fusion_forward(img_l, img_g, txt_l, txt_g)
            edl_matrix = self.CARZero_model.edl_head(torch.cat([i2t_feats, t2i_feats], dim=-1)).squeeze(-1) # (B, T)        
            edl_loss, edl_acc = self.edl_forward(i2t_feats, t2i_feats, batch["label"], split=split)
        
        return i2t_cls, t2i_cls, edl_matrix, edl_loss, edl_acc
        
    def metrics(self, batch, split):
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)
        
        i2t_cls, t2i_cls, edl_cls, edl_loss, edl_acc = self.inference(batch, split=split)
        all_logits = (i2t_cls + t2i_cls) / 2
        pos_logits = all_logits[:, :14]
        neg_logits = all_logits[:, 14:]
        
        pos_evd = edl_cls[:,:14]
        neg_evd = edl_cls[:,14:]
        alpha_pos = F.softplus(pos_evd) + 1
        alpha_neg = F.softplus(neg_evd) + 1
        
        S = alpha_pos + alpha_neg
        
        pos_probs  = alpha_pos / S                             # 질환 존재 확률
        neg_probs  = alpha_neg / S                             # 질환 부재 확률
        
        U = 2 / S # (N, 14)
        
        pred_alpha = alpha_pos > alpha_neg
        failure_case = pred_alpha != labels[:,:-1].bool() # (N, 14) 각 클래스별로 실패한 경우
        failure_detection_aurocs = []
        for i in range(14):
            failure_detection_auroc = roc_auc_score(failure_case[:, i].cpu().numpy(), U[:, i].cpu().numpy())
            self.log(f"{split}/FD_AUROC_{self.class_names[i]}", failure_detection_auroc, prog_bar=True, sync_dist=True)
            failure_detection_aurocs.append(failure_detection_auroc)
        failure_detection_mean_auroc = np.mean(failure_detection_aurocs)
        self.log(f"{split}/FD_AUROC", failure_detection_mean_auroc, prog_bar=True, sync_dist=True)
        
        U_mean = U.mean(dim=0)                                 # 클래스별 평균 불확실성
        
        targets = labels[:,:-1].int()                             # 질환 존재 → 1
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1
        
        preds = pos_probs > neg_probs

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_logits,  targets)
        self.neg_auroc_metric.update(neg_logits, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, targets.float())
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        
        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc   = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/edl_loss",       edl_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/edl_acc",        edl_acc,      prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
        self.log_dict({f"{split}/Uncertainty_{c}":     U_mean[i]
                          for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
                   
            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
            self.print(" - Uncertainty Mean:")
            uncertainty_metrics = {k: v for k, v in metrics.items() if k.startswith("val/Uncertainty_")}
            for key in uncertainty_metrics:
                class_name = key.replace("val/Uncertainty_", "")
                self.print(f"    {class_name:<22}: {uncertainty_metrics[key].item():.4f}")
                
            self.print(f" - Failure Detection Mean AUROC")
            FD_metrics = {k: v for k, v in metrics.items() if k.startswith("val/FD_AUROC_") and k != "val/FD_AUROC"}
            self.print(" - Failure Detection AUROC by Class:")
            for key in FD_metrics:
                class_name = key.replace("val/FD_AUROC_", "")
                self.print(f"    {class_name:<22}: {FD_metrics[key].item():.4f}")
                
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
            
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        
# class MCQEDLDQNWOSAMLPGLModel(LightningModule):
#     def __init__(self, cfg):
#         super().__init__()

#         self.cfg = cfg
#         self.save_hyperparameters(self.cfg)
#         self.CARZero_model = None
#         self.lr = cfg.lightning.trainer.lr
#         self.dm = None
#         self.auroc_metric = MultilabelAUROC(num_labels=14, average=None)
#         self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
#         self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
#         self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
#             'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']
        
#         self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
#         self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        
#         self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
#         self.prompts = [f"There is {cls.replace('_', ' ')} but no {neg_cls.replace('_', ' ')}." for cls in self.class_names[:-1] for neg_cls in self.class_names[:-1] if cls != neg_cls] + self.prompts 
        
#         self.pos_prompts = [f"There is {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
#         self.neg_prompts = [f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
 
#     def setup(self, stage=None):
#         if self.CARZero_model is None:
#             self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
#             self.freeze_module()
#             self.print("CARZero model loaded and frozen.")
#         if self.cfg.peft.enabled :
#             self.print("Setting up PEFT for the student model...")
#             self.set_peft()
#         if self.dm is None:
#             self.dm = self.trainer.datamodule
            
#     def set_peft(self):
#         r = self.cfg.peft.r
#         alpha = self.cfg.peft.alpha
#         dropout = self.cfg.peft.dropout
#         adaptor_name = self.cfg.peft.adaptor_name
        
#         self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")

#         # if adaptor_name == "lora":
#         #     apply_lora(
#         #         self.CARZero_model, 
#         #         r=r, 
#         #         alpha=alpha, 
#         #         dropout=dropout, 
#         #         merge_weights=False
#         #     )
#         #     self.print("LoRA adapters applied to the CARZero model.")
    
#     def freeze_module(self):
#         freeze_dict = getattr(self.cfg, "freeze", {})
#         if freeze_dict.get("image", False):
#             for param in self.CARZero_model.img_encoder.parameters():
#                 param.requires_grad = False
#         if freeze_dict.get("text", False):
#             for param in self.CARZero_model.text_encoder.parameters():
#                 param.requires_grad = False
#         if freeze_dict.get("fusion", False):
#             if self.cfg.model.CARZero.multi == False:
#                 for param in self.CARZero_model.fusion_module.parameters():
#                     param.requires_grad = False
#             else :
#                 for param in self.CARZero_model.i2t_fusion_module.parameters():
#                     param.requires_grad = False
#                 for param in self.CARZero_model.t2i_fusion_module.parameters():
#                     param.requires_grad = False

#         self.print("==== Frozen Modules ====")
#         self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
#         self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
#         if self.cfg.model.CARZero.multi == False:
#             self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
#         else :
#             self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
#             self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
#     def configure_optimizers(self):
#         optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
#         scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
#         return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
#     def i2t_forward(self, batch):
#         i2t_cls, t2i_cls = self.CARZero_model.i2t_mcq_forward(batch, i2t_only=self.cfg.model.CARZero.single_path)
        
#         targets = batch["answer_idx"].to(self.device)
        
#         logits = i2t_cls
        
#         N = logits.size(0)

#         alpha = torch.exp(logits) + 1 # (N, T)
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         probs = alpha / S
        
#         alpha_y = alpha[torch.arange(N, device=self.device), targets].unsqueeze(1)  # (N,1)
#         loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()
        
#         y = F.one_hot(targets, num_classes=logits.size(1)).to(alpha.dtype)  # (N, T)
#         tilde_alpha = y + (1.0 - y) * alpha  # (N, T)
        
#         loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
        
#         epoch = getattr(self, "current_epoch", 0)
#         lam = min(1.0, float(epoch) / self.cfg.train.lam)
#         lam = torch.tensor(lam, device=self.device, dtype=alpha.dtype)

#         loss = loss_match + lam * loss_kl
#         acc = (probs.argmax(dim=1) == targets).float().mean()
        
#         return i2t_cls, t2i_cls, loss, acc
    
#     def t2i_forward(self, batch):
#         batch = build_t2i_mcq_batch(
#             batch,
#             self.tokenizer,
#             self.prompts,
#             self.class_names,
#             max_length=self.cfg.data.text.word_num,
#             num_negatives=2,
#             no_hyb=self.cfg.data.text.no_hyb
#             )
        
#         if len(batch['imgs'].shape) != 5 :
#             self.print(f"Unexpected image batch shape: {batch['imgs'].shape}")
#             return None, None, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
#         batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
#         i2t_cls, t2i_cls = self.CARZero_model.t2i_mcq_forward(batch, t2i_only=self.cfg.model.CARZero.single_path)
        
#         targets = batch["answer_idx"].to(self.device)
        
#         logits = t2i_cls
        
#         N = logits.size(0)

#         alpha = torch.exp(logits) + 1 # (N, T)
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         probs = alpha / S
        
#         alpha_y = alpha[torch.arange(N, device=self.device), targets].unsqueeze(1)  # (N,1)
#         loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()
        
#         y = F.one_hot(targets, num_classes=logits.size(1)).to(alpha.dtype)  # (N, T)
#         tilde_alpha = y + (1.0 - y) * alpha  # (N, T)
        
#         loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
        
#         epoch = getattr(self, "current_epoch", 0)
#         lam = min(1.0, float(epoch) / 15.0)
#         lam = torch.tensor(lam, device=self.device, dtype=alpha.dtype)

#         loss = loss_match + lam * loss_kl
#         acc = (probs.argmax(dim=1) == targets).float().mean()
        
#         return i2t_cls, t2i_cls, loss, acc

#     def training_step(self, batch, batch_idx):
#         loss = self.shared_step(batch, "train")
#         return loss

#     def validation_step(self, batch, batch_idx):
#         #loss = self.shared_step(batch, "val")
#         bce_loss = self.metrics(batch, "val")
#         return {
#             #"val/loss": loss.detach(),
#             "val/bce_loss": bce_loss.detach(),
#             # "mean_auroc": mean_auroc.detach(),
#             # "class_auroc": class_auroc.detach()
#         }
    
#     def test_step(self, batch, batch_idx):
#         loss = self.shared_step(batch, "test")
#         bce_loss = self.metrics(batch, "test")
#         return {
#             "test/loss": loss.detach(),
#             "test/bce_loss": bce_loss.detach(),
#         }

#     def shared_step(self, batch, split):
#         weight = self.cfg.train.loss_weight
        
#         i2t_logits_i2t, t2i_logits_i2t, i2t_loss, i2t_acc = self.i2t_forward(batch)
#         i2t_logits_t2i, t2i_logits_t2i, t2i_loss, t2i_acc = self.t2i_forward(batch)

#         ce_loss = weight * i2t_loss + (1 - weight) * t2i_loss
        
#         self.log_dict({f"{split}/loss": ce_loss,
#                        f"{split}/i2t_loss": i2t_loss,
#                        f"{split}/t2i_loss": t2i_loss,
#                        f"{split}/i2t_acc": i2t_acc,
#                        f"{split}/t2i_acc": t2i_acc},
#                   prog_bar=True, on_epoch=True)
                 
#         return ce_loss
        
#     def metrics(self, batch, split):
#         imgs   = batch["imgs"].to(self.device)
#         labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

#         # ---------- Positive-prompt similarity ----------
#         pos_text = self.CARZero_model.process_text(
#             self.pos_prompts, self.device)
#         pos_logits = CARZero.dqn_shot_classification(
#             self.CARZero_model, imgs, pos_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi)
#         pos_logits = torch.tensor(pos_logits, device=self.device)
#         alpha_pos = torch.exp(pos_logits) + 1 

#         # ---------- Negative-prompt similarity ----------  
#         neg_text = self.CARZero_model.process_text(
#             self.neg_prompts, self.device)
#         neg_logits = CARZero.dqn_shot_classification(
#             self.CARZero_model, imgs, neg_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi) # (N, 14)
#         neg_logits = torch.tensor(neg_logits, device=self.device)
#         alpha_neg = torch.exp(neg_logits) + 1 
        
#         S = alpha_pos + alpha_neg
        
#         pos_probs  = alpha_pos / S                             # 질환 존재 확률
#         neg_probs  = alpha_neg / S                             # 질환 부재 확률
        
#         U = 2 / S # (N, 14)
        
#         U_mean = U.mean(dim=0)                                 # 클래스별 평균 불확실성
        
#         targets = labels[:,:-1].int()                             # 질환 존재 → 1
#         neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

#         # ---------- 메트릭 누적 ----------
#         self.auroc_metric.update(pos_probs,  targets)
#         self.neg_auroc_metric.update(neg_probs, neg_targets)

#         # ---------- BCE 손실 (positive-prompt 기준) ----------
#         pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, targets.float())
#         neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
#         #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
#         weight = self.cfg.train.loss_weight
#         bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

#         # ---------- 지표 집계 ----------
#         class_auroc      = self.auroc_metric.compute()
#         neg_class_auroc  = self.neg_auroc_metric.compute()
#         pos_mean_auroc       = class_auroc.mean()
#         neg_mean_auroc   = neg_class_auroc.mean()
#         mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

#         # ---------- 로깅 ----------
#         self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
#         self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
#         self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
#         self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

#         # 클래스별 AUROC도 한꺼번에 로깅
#         self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
#                        for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
#         self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
#                        for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
#         self.log_dict({f"{split}/Uncertainty_{c}":     U_mean[i]
#                           for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)

#         return bce_loss
        
#     def on_validation_epoch_end(self):
#         metrics = self.trainer.callback_metrics

#         if self.trainer.is_global_zero:
#             self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
                   
#             # 기본 손실 및 평균 AUROC 출력
#             for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
#                 if key in metrics:
#                     self.print(f" - {key:<17}: {metrics[key].item():.4f}")

#             # 클래스별 AUROC만 따로 정렬 출력
#             self.print(" - Class-wise AUROC:")
#             class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
#             for key in class_metrics:
#                 class_name = key.replace("val/auroc_", "")
#                 self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
#             self.print(" - Negative Class-wise AUROC:")
#             neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
#             for key in neg_class_metrics:
#                 class_name = key.replace("val/neg_auroc_", "")
#                 self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
#     def on_test_epoch_end(self):
#         class_auroc = self.auroc_metric.compute()
#         mean_auroc = torch.mean(class_auroc)

#         if self.trainer.is_global_zero:
#             self.print(f"[TEST] Epoch Summary:")
#             self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
#             for i, cls in enumerate(self.class_names[:-1]):
#                 self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
#             self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
#             neg_class_auroc = self.neg_auroc_metric.compute()
#             for i, cls in enumerate(self.class_names[:-1]):
#                 self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

#         # metric 초기화 (다음 test run 대비)
#         self.auroc_metric.reset()
#         self.neg_auroc_metric.reset()


class MultiLabelMCQDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
            self.freeze_module()
        # if self.cfg.peft.enabled :
        #     self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.CARZero_model.fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def _forward_one_prompt(self, imgs, txt_ids, txt_mask, txt_type):
        """
        imgs      : (B, C, H, W)
        txt_ids   : (B, L)
        txt_mask  : (B, L)
        txt_type  : (B, L) or None
        반환      : img_cls (B, D), txt_cls (B, D)
        """
        forward_dict = {
            "imgs": imgs,
            "caption_ids": txt_ids,
            "attention_mask": txt_mask
        }
        if txt_type is not None:
            forward_dict["token_type_ids"] = txt_type
        *_, i2t_cls, t2i_cls = self.CARZero_model(forward_dict)   # (B,), (B,)
        return i2t_cls, t2i_cls

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            "val_loss": loss.detach(),
            "val_bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test_loss": loss.detach(),
            "test_bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        ans_idx = batch["answer_idx"].to(self.device)      # (B,)
        imgs     = batch["imgs"].to(self.device)           # (B,3,H,W)
        targets = batch["targets"].to(self.device)           # (B,3)
        ids_all  = batch["caption_ids_all"].to(self.device)      # (B,4,L)
        mask_all = batch["attention_mask_all"].to(self.device)
        type_all = batch.get("token_type_ids_all", None)
        
        if type_all is not None:
            type_all = type_all.to(self.device)

        B, N, L = ids_all.shape            # N = 4

        i2t_list, t2i_list = [], []

        # ───────── 4-프롬프트 순차 처리 ──────────
        for j in range(N):
            txt_ids  = ids_all[:, j, :]        # (B,L)
            txt_mask = mask_all[:, j, :]
            txt_type = type_all[:, j, :] if type_all is not None else None

            i2t_cls, t2i_cls = self._forward_one_prompt(imgs, txt_ids, txt_mask, txt_type)

            i2t_list.append(i2t_cls)        # 정답 InfoNCE용
            t2i_list.append(t2i_cls)

        i2ts = torch.stack(i2t_list, 1)   # (B,4,D)
        t2is = torch.stack(t2i_list, 1)   # (B,

        pos_img = i2ts[torch.arange(B), ans_idx]   # (B,D)
        pos_txt = t2is[torch.arange(B), ans_idx]   # (B,D

        info_loss = self.CARZero_model.calc_loss(pos_img, pos_txt)
        
        row_idx = torch.arange(B, device=self.device)
        i2t_logits = i2ts[row_idx, :, row_idx]
        bce_loss = F.binary_cross_entropy_with_logits(
            i2t_logits, targets, reduction="mean"
        )
        
        w = self.cfg.train.loss_weight
        loss = w* info_loss + (1 - w) * bce_loss
        
        self.log_dict({f"{split}_loss": loss,
                   f"{split}_info": info_loss,
                   f"{split}_bce": bce_loss},
                  prog_bar=True, on_epoch=True)
                 
        return loss
        
    def metrics(self, batch, split):
        processes_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        similarity = CARZero.dqn_shot_classification(
            self.CARZero_model,
            batch["imgs"].to(self.device),
            processes_text,).values
        similarity = torch.tensor(similarity).to(self.device)
        labels = batch["label"].to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        probs = torch.sigmoid(similarity)
        
        self.auroc_metric.update(probs, labels.int())  # 배치 단위로 누적만
    
        self.log(f"{split}_bce_loss", loss,
             on_epoch=True, prog_bar=True)
        
        return loss
        
    def on_validation_epoch_end(self):
        class_auroc = self.auroc_metric.compute()       # ✅ epoch 전체 기반
        mean_auc    = class_auroc.mean()

        # 로그 및 출력
        self.log("val_mean_auroc", mean_auc, prog_bar=True)
        for i, cls in enumerate(self.dm.train_dataset.class_names):
            self.log(f"val_auroc_{cls}", class_auroc[i])

        if self.trainer.is_global_zero:
            print(f"[VAL] Epoch {self.current_epoch} Summary:")
            print(f" - val_mean_auroc : {mean_auc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
        
        self.auroc_metric.reset()                       # 다음 epoch 위해 초기화
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        self.log("test_mean_auroc", mean_auroc, prog_bar=True)
        for i, cls in enumerate(self.dm.train_dataset.class_names):
            self.log(f"test_auroc_{cls}", class_auroc[i])
            
        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        
def apply_lora(model: torch.nn.Module,
               target_keywords=("query", "key", "value", "dense", "qkv", "proj", "linear"),
               r: int = 8,
               alpha: int = 64,
               dropout: float = 0.05,
               merge_weights: bool = False):

    for name, child in model.named_children():
        # 재귀 탐색
        apply_lora(child, target_keywords, r, alpha, dropout, merge_weights)

        # Linear 조건 매칭
        if isinstance(child, torch.nn.Linear) and any(k in name for k in target_keywords):
            lora = LoRALinear(
                child,
                adapter_name="lora",
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                merge_weights=merge_weights,
            )
            # base weight freeze
            for p in lora.base_layer.parameters():
                p.requires_grad = False
            setattr(model, name, lora)

class LORAKDMCQCARZEROModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.teacher = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
        self.student = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
        
        self.teacher.eval().requires_grad_(False)
        self.student.train().requires_grad_(True)
        
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)

    def setup(self, stage=None):
        if self.cfg.peft.enabled :
            self.print("Setting up PEFT for the student model...")
            self.set_peft()
            
        self.freeze_module()
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def set_peft(self):
        r = self.cfg.peft.r
        alpha = self.cfg.peft.alpha
        dropout = self.cfg.peft.dropout
        adaptor_name = self.cfg.peft.adaptor_name
        
        self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")

        if adaptor_name == "lora":
            apply_lora(
                self.student, 
                r=r, 
                alpha=alpha, 
                dropout=dropout, 
                merge_weights=False
            )
            self.print("LoRA adapters applied to the student model.")
        
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.student.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.student.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.student.fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.student.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.student.text_encoder.parameters()))
        self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.student.fusion_module.parameters()))

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.student)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def _forward_one_prompt(self, imgs, txt_ids, txt_mask, txt_type):
        """
        imgs      : (B, C, H, W)
        txt_ids   : (B, L)
        txt_mask  : (B, L)
        txt_type  : (B, L) or None
        반환      : img_cls (B, D), txt_cls (B, D)
        """
        forward_dict = {
            "imgs": imgs,
            "caption_ids": txt_ids,
            "attention_mask": txt_mask
        }
        if txt_type is not None:
            forward_dict["token_type_ids"] = txt_type
        *_, i2t_cls, t2i_cls = self.student(forward_dict)   # (B,), (B,)
        return i2t_cls, t2i_cls

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            "val_loss": loss.detach(),
            "val_bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test_loss": loss.detach(),
            "test_bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        ans_idx = batch["answer_idx"].to(self.device)      # (B,)
        imgs     = batch["imgs"].to(self.device)           # (B,3,H,W)
        ids_all  = batch["caption_ids_all"].to(self.device)      # (B,4,L)
        mask_all = batch["attention_mask_all"].to(self.device)
        type_all = batch.get("token_type_ids_all", None)
        if type_all is not None:
            type_all = type_all.to(self.device)

        B, N, L = ids_all.shape            # N = 4

        batch_arange = torch.arange(B, device=self.device)       # (B,)

        sel_ids  = ids_all[batch_arange, ans_idx]          # (B, L)
        sel_mask = mask_all[batch_arange, ans_idx]          # (B, L)
        sel_type = (
        type_all[batch_arange, ans_idx] if type_all is not None else None
        )  # (B, L) or None      
        _, t_img_emb_g = self.teacher.image_encoder_forward(imgs)
        _, t_txt_emb_g, _ = self.teacher.text_encoder_forward(sel_ids, sel_mask, sel_type)

        _, s_img_emb_g = self.student.image_encoder_forward(imgs)
        _, s_txt_emb_g, _ = self.student.text_encoder_forward(sel_ids, sel_mask, sel_type)

        kd_loss = F.mse_loss(t_img_emb_g, s_img_emb_g) + F.mse_loss(t_txt_emb_g, s_txt_emb_g)
        
        i2t_list, t2i_list = [], []

        # ───────── 4-프롬프트 순차 처리 ──────────
        for j in range(N):
            txt_ids  = ids_all[:, j, :]        # (B,L)
            txt_mask = mask_all[:, j, :]
            txt_type = type_all[:, j, :] if type_all is not None else None

            i2t_cls, t2i_cls = self._forward_one_prompt(imgs, txt_ids, txt_mask, txt_type)

            i2t_list.append(i2t_cls)        # 정답 InfoNCE용
            t2i_list.append(t2i_cls)

        i2ts = torch.stack(i2t_list, 1)   # (B,4,D)
        t2is = torch.stack(t2i_list, 1)   # (B,4,D)

        pos_img = i2ts[torch.arange(B), ans_idx]   # (B,D)
        pos_txt = t2is[torch.arange(B), ans_idx]   # (B,D)

        info_loss = self.student.calc_loss(pos_img, pos_txt)
        
        row_idx = torch.arange(B, device=self.device)
        i2t_logits = i2ts[row_idx, :, row_idx]
        ce_loss = _infonce_bidir(
            i2t_logits, ans_idx, reduction="mean"
        )
        
        w = self.cfg.train.loss_weight
        loss = w*info_loss + (1 - w) * ce_loss + self.cfg.train.kd_weight * kd_loss
        
        self.log_dict({f"{split}_loss": loss,
                   f"{split}_info": info_loss,
                   f"{split}_ce": ce_loss,
                   f"{split}_kd": kd_loss},
                  prog_bar=True, on_epoch=True)
                 
        return loss
        
    def metrics(self, batch, split):
        processes_text = self.student.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        similarity = CARZero.dqn_shot_classification(
            self.student,
            batch["imgs"].to(self.device),
            processes_text,).values
        similarity = torch.tensor(similarity).to(self.device)
        labels = batch["label"].to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        probs = torch.sigmoid(similarity)
        
        self.auroc_metric.update(probs, labels.int())  # 배치 단위로 누적만
    
        self.log(f"{split}_bce_loss", loss,
             on_epoch=True, prog_bar=True)
        
        return loss
        
    def on_validation_epoch_end(self):
        class_auroc = self.auroc_metric.compute()       # ✅ epoch 전체 기반
        mean_auc    = class_auroc.mean()

        # 로그 및 출력
        self.log("val_mean_auroc", mean_auc, prog_bar=True)
        for i, cls in enumerate(self.dm.train_dataset.class_names):
            self.log(f"val_auroc_{cls}", class_auroc[i])

        if self.trainer.is_global_zero:
            print(f"[VAL] Epoch {self.current_epoch} Summary:")
            print(f" - val_mean_auroc : {mean_auc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
        
        self.auroc_metric.reset()                       # 다음 epoch 위해 초기화
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        self.log("test_mean_auroc", mean_auroc, prog_bar=True)
        for i, cls in enumerate(self.dm.train_dataset.class_names):
            self.log(f"test_auroc_{cls}", class_auroc[i])
            
        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        
class TextAugmentCLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        # 모델 · 데이터모듈은 setup()에서 불러옵니다.
        self.CARZero_model = None
        self.dm            = None
        self.lr            = cfg.lightning.trainer.lr

        # AUROC 메트릭 (positive / negative)
        self.auroc_metric     = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=15, average=None)

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
            self.freeze_module()
        # if self.cfg.peft.enabled :
        #     self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.CARZero_model.fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        
    def set_peft(self):
        r = self.cfg.peft.r
        alpha = self.cfg.peft.alpha
        dropout = self.cfg.peft.dropout
        adaptor_name = self.cfg.peft.adaptor_name
        
        self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")

        if adaptor_name == "lora":
            apply_lora(
                self.CARZero_model, 
                r=r, 
                alpha=alpha, 
                dropout=dropout, 
                merge_weights=False
            )
            self.print("LoRA adapters applied to the CARZero model.")

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc = self.metrics(batch, "val")
        return {
            "val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            "val/mean_auroc": mean_auroc.detach(),
            "val/class_auroc": class_auroc.detach(),
            "val/neg_mean_auroc": neg_mean_auroc.detach(),
            "val/neg_class_auroc": neg_class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
            "test/mean_auroc": mean_auroc.detach(),
            "test/class_auroc": class_auroc.detach(),
            "test/neg_mean_auroc": neg_mean_auroc.detach(),
            "test/neg_class_auroc": neg_class_auroc.detach()
        }

    def shared_step(self, batch, split):
        """Similar to traning step"""
        _,_,_,_,_, i2t_cls, t2i_cls = self.CARZero_model(batch)
        loss = self.CARZero_model.calc_loss_aug(
            i2t_cls, t2i_cls, batch["truth"],
        )

        self.log(
            f"{split}/loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        
        return loss
    
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text).values
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text).values
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)

        return bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc
    
    def on_train_epoch_start(self):
        # update epoch-dependent probabilities inside the shared collator
        if hasattr(self.trainer.datamodule, "train_collate"):
            self.trainer.datamodule.train_collate.set_epoch(self.current_epoch)
            prob_cfg = self.trainer.datamodule.train_collate.prob_cfg
            log_dict = {
                "train/nf"    : float(prob_cfg["neg_nf"][0]),      # NF 선택 확률
                "train/nf_neg"   : float(prob_cfg["neg_nf"][1]),      # NEG 선택 확률
                "train/pos"      : float(prob_cfg["pos_neg_hyb"][0]), # POS
                "train/neg"      : float(prob_cfg["pos_neg_hyb"][1]), # NEG
                "train/hyb"      : float(prob_cfg["pos_neg_hyb"][2]), # HYB
                "train/spec"     : float(prob_cfg["sub_spec_abn"][0]),# SPEC
                "train/abn"      : float(prob_cfg["sub_spec_abn"][1]) # ABN
            }

        # ③ log_dict 호출
        # on_step=False → step 단위 누적 없음, on_epoch=True → epoch 끝에 기록
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")

    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()

def build_signed_target(batch, class_names):
    """
    반환: T ∈ ℤ^{B×P}
      T[b, p] =  +(c+1)  if labels[b,c]==1 and prompt p is POS for class c (c!=NF)
                =  -(c+1)  if labels[b,c]==1 and prompt p is NEG for class c (c!=NF)
                =   0      otherwise
    """
    labels = batch['labels']                  # [B,C]
    p2c    = batch['prompt_target_idx']       # list[int], len=P
    ptype  = batch['prompt_type']             # list[str], len=P
    B, C   = labels.shape
    P      = len(p2c)
    device = labels.device
    NF     = class_names.index('No Finding') if 'No Finding' in class_names else -1

    T = torch.zeros((B, P), dtype=torch.int16, device=device)

    for p in range(P):
        c = p2c[p]
        if c == NF:                # NF 프롬프트는 패스
            continue
        pos_mask = (labels[:, c] > 0.5)
        if ptype[p] == 'pos':
            T[pos_mask, p] =  (c + 1)
        elif ptype[p] == 'neg':
            T[pos_mask, p] = -(c + 1)
        elif ptype[p] == 'both':
            # 스키마상 생기지 않는다면 무시. 필요 시 아래처럼 한쪽으로만 표기
            # T[pos_mask, p] =  (c + 1)
            pass
    return T

def signed_margin_loss(T: torch.Tensor,
                       i2t_cls: torch.Tensor,
                       t2i_cls: torch.Tensor | None = None,
                       margin: float = 0.2):
    """
    T[b,p] = +(c+1) (양성-긍정프롬프트),  -(c+1) (양성-부정프롬프트), 0 (기타)
    i2t_cls, t2i_cls : [B,P] 또는 [P,B]  (자동 전치)
    반환: (loss_total, loss_i2t, loss_t2i or None)
    """
    B, P = T.shape

    def to_BP(S):
        if S.shape == (B, P): return S
        if S.shape == (P, B): return S.T
        raise ValueError(f"Shape mismatch: got {tuple(S.shape)}, expected {(B,P)} or {(P,B)}")

    def one_dir(S_BP: torch.Tensor):
        # 연산 기준 장치/dtype로 모두 맞추기
        S = to_BP(S_BP).to(dtype=torch.float32)
        dev = S.device
        T_loc = T.to(dev)

        # 배치에 등장한 클래스 수(C=최대 라벨)
        C = int(T_loc.abs().max().item())
        if C == 0:
            return torch.tensor(0.0, device=dev)

        # 클래스별 마스크 (모두 S.device에서 생성)
        Tp   = T_loc.unsqueeze(1)                          # [B,1,P]
        cls  = torch.arange(1, C+1, device=dev).view(1, C, 1)  # [1,C,1]
        Mpos = (Tp ==  cls)                                # [B,C,P]
        Mneg = (Tp == -cls)                                # [B,C,P]

        NEG_INF = torch.finfo(S.dtype).min
        sB1P = S.unsqueeze(1)                              # [B,1,P]
        s_pos = sB1P.masked_fill(~Mpos, NEG_INF).amax(dim=2)  # [B,C]
        s_neg = sB1P.masked_fill(~Mneg, NEG_INF).amax(dim=2)  # [B,C]

        valid = Mpos.any(dim=2) & Mneg.any(dim=2)          # 둘 다 있어야 유효
        hinge = F.relu(margin - (s_pos - s_neg))           # [B,C]
        loss  = hinge[valid].mean() if valid.any() else torch.tensor(0.0, device=dev)
        return loss

    loss_i2t = one_dir(i2t_cls)
    loss_t2i = one_dir(t2i_cls) if t2i_cls is not None else None
    loss_tot = (loss_i2t + loss_t2i) * 0.5 if loss_t2i is not None else loss_i2t
    return loss_tot

def _infonce_row(S: torch.Tensor, mask: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    행 단위 InfoNCE (멀티-포지티브): L = logsumexp(all/tau) - logsumexp(pos/tau)
    S   : [N, M] (유사도/로짓)
    mask: [N, M] (True=양성)
    """
    S = (S.to(torch.float32) / tau)
    mask = mask.to(dtype=torch.bool, device=S.device)
    NEG_INF = torch.finfo(S.dtype).min

    lse_all = torch.logsumexp(S, dim=1)                       # [N]
    lse_pos = torch.logsumexp(S.masked_fill(~mask, NEG_INF), dim=1)  # [N]

    valid = mask.any(dim=1)                                   # 양성이 있는 행만
    return (lse_all[valid] - lse_pos[valid]).mean() if valid.any() else torch.zeros((), device=S.device)

def _infonce_bidir(S: torch.Tensor, mask: torch.Tensor, tau: float=0.07) -> torch.Tensor:
    """
    양방향 InfoNCE 평균:
      - 행방향: i2t_sub vs mask_sub
      - 열방향: t2i_sub vs mask_sub.T
    i2t_sub: [B, P_sel], t2i_sub: [P_sel, B], mask_sub: [B, P_sel]
    """
    L_row = _infonce_row(S, mask, tau)
    L_col = _infonce_row(S.T, mask.T, tau)
    print(f"InfoNCE row loss: {L_row.item()}, col loss: {L_col.item()}")

    return 0.5 * (L_row + L_col)

def _mcq_loss(S: torch.Tensor, target_idx: torch.Tensor, reduction: str="mean") -> torch.Tensor:
    L_row = torch.nn.functional.cross_entropy(S, target_idx, reduction=reduction)

    return L_row

class MarginCONCLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        # 모델 · 데이터모듈은 setup()에서 불러옵니다.
        self.CARZero_model = None
        self.dm            = None
        self.lr            = cfg.lightning.trainer.lr

        # AUROC 메트릭 (positive / negative)
        self.auroc_metric     = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=15, average=None)

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
            self.freeze_module()
        # if self.cfg.peft.enabled :
        #     self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.CARZero_model.fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch):
        loss = self.shared_step(batch, "val")
        bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc = self.metrics(batch, "val")
        return {
            "val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            "val/mean_auroc": mean_auroc.detach(),
            "val/class_auroc": class_auroc.detach(),
            "val/neg_mean_auroc": neg_mean_auroc.detach(),
            "val/neg_class_auroc": neg_class_auroc.detach()
        }
    
    def test_step(self, batch):
        loss = self.shared_step(batch, "test")
        bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
            "test/mean_auroc": mean_auroc.detach(),
            "test/class_auroc": class_auroc.detach(),
            "test/neg_mean_auroc": neg_mean_auroc.detach(),
            "test/neg_class_auroc": neg_class_auroc.detach()
        }

    def shared_step(self, batch, split):
        """Similar to traning step"""
        _,_,_,_,_, i2t_cls, t2i_cls = self.CARZero_model(batch)
        loss, pos_loss, neg_loss, margin_loss = self.loss(batch, i2t_cls, t2i_cls)

        self.log(
            f"{split}/loss",
            loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        
        self.log(
            f"{split}/pos_loss",
            pos_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        
        self.log(
            f"{split}/neg_loss",
            neg_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        
        self.log(
            f"{split}/margin_loss",
            margin_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        
        return loss
    
    def loss(self, batch, i2t_cls, t2i_cls):
        """
        배치에 대한 손실 계산
        :param batch: 배치 데이터
        :param i2t_cls: 이미지 → 텍스트 클래스 예측
        :param t2i_cls: 텍스트 → 이미지 클래스 예측
        :return: 손실 값
        """
        
        pos_idx  = [i for i, t in enumerate(batch['prompt_type']) if t == 'pos']
        neg_idx  = [i for i, t in enumerate(batch['prompt_type']) if t == 'neg']
        
        i2t_cls_pos = i2t_cls[:, pos_idx]
        t2i_cls_pos = t2i_cls[pos_idx, :]
        i2t_cls_neg = i2t_cls[:, neg_idx]
        t2i_cls_neg = t2i_cls[neg_idx, :]
        
        truth_pos = batch['truth'][:, pos_idx]
        truth_neg = batch['truth'][:, neg_idx]
        truth_pos = truth_pos.float().to(self.device)
        truth_neg = truth_neg.float().to(self.device)
        
        # ---------- Positive-prompt 손실 ----------
        i2t_loss_pos = _infonce_bidir(i2t_cls_pos, truth_pos)
        t2i_loss_pos = _infonce_bidir(t2i_cls_pos, truth_pos.T)
        pos_loss = 0.5 * (i2t_loss_pos + t2i_loss_pos)
        
        # ---------- Negative-prompt 손실 ----------
        i2t_loss_neg = _infonce_bidir(i2t_cls_neg, truth_neg)
        t2i_loss_neg = _infonce_bidir(t2i_cls_neg, truth_neg.T)
        neg_loss = 0.5 * (i2t_loss_neg + t2i_loss_neg)
        
        # ---------- Signed Margin Loss ----------
        T = build_signed_target(batch, self.dm.train_dataset.class_names)
        margin_loss = signed_margin_loss(T, i2t_cls, t2i_cls)
        
        # ---------- Total Loss ----------
        total_loss = self.cfg.train.pos*pos_loss + self.cfg.train.neg*neg_loss + self.cfg.train.margin*margin_loss

        return total_loss, pos_loss, neg_loss, margin_loss
    
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["labels"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text).values
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text).values
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)})
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)})
        
        return bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")

    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        
def _dcl_row_multipositive(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    tau: float = 0.07,
    reduce: str = "mean",
    ignore_mask: torch.Tensor | None = None,  # ← 추가
) -> torch.Tensor:
    """
    행 기준 멀티-포지티브 DCL:
      L_r = -(1/K_r) * sum_{j in Pos(r)} (x[r,j]) + logsumexp_{j in Neg(r)} (x[r,j]),
      x = logits / tau.

    추가: ignore_mask가 True인 위치는 포지티브/네거티브 모두에서 '완전히 제외'
          (즉, 손실에 기여하지 않음).
    """
    assert logits.shape == pos_mask.shape, "logits/pos_mask 형상이 일치해야 하옵니다."
    x = logits.to(torch.float32) / tau
    pos_mask = pos_mask.to(dtype=torch.bool, device=x.device)

    if ignore_mask is None:
        ignore_mask = torch.zeros_like(pos_mask, dtype=torch.bool, device=x.device)
    else:
        ignore_mask = ignore_mask.to(dtype=torch.bool, device=x.device)

    # 네거티브는 '포지티브 또는 무시'가 아닌 위치
    neg_mask = ~(pos_mask | ignore_mask)

    # 행별 포지티브/네거티브 개수 및 유효 행
    K = (pos_mask & ~ignore_mask).sum(dim=1)        # [R]
    M = neg_mask.sum(dim=1)                         # [R]
    valid = (K > 0) & (M > 0)
    if not valid.any():
        return logits.new_tensor(0.0, dtype=torch.float32, device=x.device)

    x = x[valid]
    pos_mask = pos_mask[valid]
    neg_mask = neg_mask[valid]
    K = K[valid]

    # 포지티브 평균항
    pos_sum  = x.masked_fill(~pos_mask, 0.0).sum(dim=1)     # [Rv]
    pos_mean = pos_sum / torch.clamp(K, min=1)              # [Rv]

    # 네거티브 LSE 항
    NEG_INF = torch.finfo(x.dtype).min
    neg_x   = x.masked_fill(~neg_mask, NEG_INF)             # [Rv, C]
    neg_lse = torch.logsumexp(neg_x, dim=1)                 # [Rv]

    loss_row = -(pos_mean) + neg_lse                        # [Rv]

    if reduce == "mean":
        return loss_row.mean()
    elif reduce == "sum":
        return loss_row.sum()
    else:
        raise ValueError("reduce는 'mean' 또는 'sum'만 지원하옵니다.")
        
class DCLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        # 모델 · 데이터모듈은 setup()에서 불러옵니다.
        self.CARZero_model = None
        self.dm            = None
        self.lr            = cfg.lightning.trainer.lr

        # AUROC 메트릭 (positive / negative)
        self.auroc_metric     = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
            self.freeze_module()
        # if self.cfg.peft.enabled :
        #     self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.CARZero_model.fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch):
        loss = self.shared_step(batch, "val")
        bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc = self.metrics(batch, "val")
        return {
            "val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            "val/mean_auroc": mean_auroc.detach(),
            "val/class_auroc": class_auroc.detach(),
            "val/neg_mean_auroc": neg_mean_auroc.detach(),
            "val/neg_class_auroc": neg_class_auroc.detach()
        }
    
    def test_step(self, batch):
        loss = self.shared_step(batch, "test")
        bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
            "test/mean_auroc": mean_auroc.detach(),
            "test/class_auroc": class_auroc.detach(),
            "test/neg_mean_auroc": neg_mean_auroc.detach(),
            "test/neg_class_auroc": neg_class_auroc.detach()
        }

    def shared_step(self, batch, split):
        """Similar to traning step"""
        _,_,_,_,_, i2t_cls, t2i_cls = self.CARZero_model(batch)
        loss, i2t_i_loss, i2t_t_loss, t2i_t_loss, t2i_i_loss = self.loss(batch, i2t_cls, t2i_cls)

        self.log(
            f"{split}/loss",
            loss,
            on_epoch=True,
            on_step=True if split == "train" else False,
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        
        self.log(
            f"{split}/i2t_i_loss",
            i2t_i_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        
        self.log(
            f"{split}/i2t_t_loss",
            i2t_t_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        
        self.log(
            f"{split}/t2i_i_loss",
            t2i_i_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        
        self.log(
            f"{split}/t2i_t_loss",
            t2i_t_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        
        return loss
    
    def loss(self, batch, i2t_cls, t2i_cls):
        device = i2t_cls.device
        truth = (batch['prompt_truth'].to(device) > 0.5)   # [B,P]
        B, P = truth.shape

        # prompt_type: pos=1, neg=0  → [P] 또는 [B,P] 모두 허용
        ptype = batch['prompt_type'].to(device)
        if ptype.ndim == 2:
            ptype = ptype[0]                      # [P]
        posP = (ptype == 1)                       # [P]
        negP = ~posP                              # [P]

        # NF 프롬프트 식별 (dataset에 넣어둔 prompt_target_idx 사용)
        tgt_idx = batch['prompt_target_idx'].to(device)     # [P]
        nf_id = self.dm.train_dataset.class_names.index('No Finding')
        is_nf_pos_prompt = (tgt_idx == nf_id) & posP        # [P]  (NF 긍정 프롬프트)

        # ----------------- 이미지 앵커(i2t_i, t2i_i): 멀티포지티브 InfoNCE + NF 제외 -----------------
        #  - 부정 프롬프트의 positive pair 제외
        #  - NF positive pair도 제외  ← 중요!
        pos_mask_i = truth & posP.view(1, P)                         # [B,P] (긍정 프롬프트의 양성)
        drop_mask  = truth & (negP.view(1, P) | is_nf_pos_prompt.view(1, P))  # [B,P] (완전 제외)
        pos_mask_i = pos_mask_i & ~drop_mask                         # 남길 진짜 포지티브

        # i2t_i: [B,P]
        i2t_i_loss = _infonce_row(i2t_cls, pos_mask_i, tau=1.0)

        # t2i_i: t2i를 [B,P]로 맞춘 뒤 동일 처리
        if t2i_cls.shape == (P, B):
            t2i_BP = t2i_cls.T
        elif t2i_cls.shape == (B, P):
            t2i_BP = t2i_cls
        else:
            raise ValueError(f"t2i_cls shape {tuple(t2i_cls.shape)} not in {{(B,P),(P,B)}}")

        t2i_i_loss = _infonce_row(t2i_BP, pos_mask_i, tau=1.0)

        # ----------------- 텍스트 앵커(i2t_t, t2i_t): DCL + 부정 프롬프트 행 제거 -----------------
        keep_rows = torch.nonzero(posP, as_tuple=False).squeeze(1)    # 긍정 프롬프트 행만 유지
        if keep_rows.numel() > 0:
            # i2t_t: i2t_cls.T -> [P,B]
            i2t_t_logits = i2t_cls.T.index_select(0, keep_rows)       # [P_pos, B]
            i2t_t_mask   = truth.T.index_select(0, keep_rows)         # [P_pos, B]
            i2t_t_loss = _dcl_row_multipositive(
                logits=i2t_t_logits, pos_mask=i2t_t_mask,
                tau=1.0, reduce="mean"
            )

            # t2i_t: t2i를 [P,B]로 맞춘 뒤
            t2i_PB = t2i_cls if t2i_cls.shape == (P, B) else t2i_cls.T
            t2i_t_logits = t2i_PB.index_select(0, keep_rows)          # [P_pos, B]
            t2i_t_mask   = truth.T.index_select(0, keep_rows)         # [P_pos, B]
            t2i_t_loss = _dcl_row_multipositive(
                logits=t2i_t_logits, pos_mask=t2i_t_mask,
                tau=1.0, reduce="mean"
            )
        else:
            i2t_t_loss = i2t_cls.new_tensor(0.0, device=device)
            t2i_t_loss = i2t_cls.new_tensor(0.0, device=device)

        # ----------------- 이미지 앵커 손실 동적 가중 (양성비 높을수록 가중 ↓) -----------------
        with torch.no_grad():
            A_plus_rows = pos_mask_i.float().mean(dim=1)          # [B]
            w_row = (1.0 - A_plus_rows).mean().clamp(min=0.1)     # 스칼라, 하한 0.1

        loss = (
            self.cfg.train.i2t_col * i2t_t_loss +
            self.cfg.train.t2i_row * t2i_t_loss +
            w_row * (self.cfg.train.i2t_row * i2t_i_loss + self.cfg.train.t2i_col * t2i_i_loss)
        )

        return loss, i2t_i_loss, i2t_t_loss, t2i_t_loss, t2i_i_loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["labels"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text).values
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text).values
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:, :14]).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)  # No Finding 제외
        
        return bce_loss, mean_auroc, class_auroc, neg_mean_auroc, neg_class_auroc
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")

    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        
class PosNegDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
        self.prompts = [f"There is {cls.replace('_', ' ')} but no {neg_cls.replace('_', ' ')}." for cls in self.class_names[:-1] for neg_cls in self.class_names[:-1] if cls != neg_cls] + self.prompts 
 
    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.CARZero_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.CARZero_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.CARZero_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls = self.CARZero_model(batch)
        loss = self.CARZero_model.calc_loss(
        i2t_cls, t2i_cls
        )

        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        
        return loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.pos_prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi)
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi) # (N, 14)
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()