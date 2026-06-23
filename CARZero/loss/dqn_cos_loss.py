## 使用 nn.BCEWithLogitsLoss() 作为loss， 对于label为-1的值，不计算loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb
import math

class DQNCOSLoss(nn.Module):
    def __init__(self):
        super(DQNCOSLoss, self).__init__()

    def forward(self, input):
        batch_size = input.size(0)
        target = Variable(torch.LongTensor(range(batch_size))).to(input.device)
        loss = 0
        loss += nn.CrossEntropyLoss()(input, target)
        loss += nn.CrossEntropyLoss()(input.transpose(1, 0), target)
        return loss / 2

class MatrixInfoNCELoss(nn.Module):
    r"""
    Symmetric (row+column) InfoNCE for similarity matrices.

    Args
    ----
    eps : float
        Numerical stabiliser.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    # ---------- 내부: 단방향(row) InfoNCE ----------
    def _row_info_nce(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = logits - logits.max(dim=-1, keepdim=True).values      # 안정화
        exp_s  = logits.exp()
        pos    = mask.bool()
        neg = ~pos

        num = (exp_s * pos).sum(dim=-1)    # (anchor,)
        #den = (exp_s * neg).sum(dim=-1)    # (anchor,)
        den = exp_s.sum(dim=-1)            # (anchor,)

        valid = pos.any(dim=-1)            # 양성 없는 anchor 제외
        if not valid.any():
            return logits.new_tensor(0.0, requires_grad=True)

        loss_vec = -torch.log((num + self.eps) / (den + self.eps))
        return loss_vec[valid].mean()

    # ---------- 전체(양방향) ----------
    def forward(self, logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if logits.shape != truth.shape:
            raise ValueError(f"Shape mismatch {logits.shape} vs {truth.shape}")

        loss_row = self._row_info_nce(logits, truth)           # 이미지→텍스트
        loss_col = self._row_info_nce(logits.t(), truth.t())             # 텍스트→이미지
        return 0.5 * (loss_row + loss_col)