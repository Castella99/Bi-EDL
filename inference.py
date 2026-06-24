#!/usr/bin/env python3
"""
Bi-EDL Inference — Classification + Uncertainty Evaluation
RiskControl.ipynb 패턴으로 backbone 직접 사용.

출력:
  1. 분류 성능  : PNC / Pos / Neg AUROC 테이블 (Inference_NIH14.py 스타일)
  2. 불확실성 비교: MSP / Energy / MaxLogit / EDL / ODIN AURC 테이블

Usage:
    python inference.py \
        --ckpt_path checkpoints/best_model.ckpt \
        --cfg_path  configs/chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_Bi_EDL.yaml \
        --data_path ../data/NIH \
        --method msp energy maxlogit edl odin
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Dict, Any, List

import CARZero
from utils import split_list, calculate_pnc_logit, calculate_metric


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural Thickening', 'Hernia',
]
# calculate_metric은 nf=False일 때 마지막 원소를 스킵 → 15번째 dummy 필요
CLASS_NAMES_15 = CLASS_NAMES + ['No Finding']

# 파인튜닝에서 사용한 prompt와 동일
_DISEASE = [
    'Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration',
    'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
    'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis',
    'Pleural Thickening', 'Hernia',
]
POS_PROMPTS = [f"There is {d}." for d in _DISEASE]    # 14
NEG_PROMPTS = [f"There is no {d}." for d in _DISEASE]  # 14
ALL_PROMPTS = POS_PROMPTS + NEG_PROMPTS                 # 28


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_backbone(ckpt_path: str, cfg, device: str):
    """
    CARZero backbone을 로드하고, Lightning checkpoint의 backbone_model.* 가중치 적용.
    MCQEDLLightModel 불필요 — RiskControl.ipynb 패턴.
    """
    backbone = CARZero.load_CARZero(
        name="CARZero_vit_b_16",
        device=device,
        multi=cfg.model.CARZero.multi,
        cfg=cfg,
    )

    ckpt_state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    prefix = "CARZero_model."
    filtered = {
        k[len(prefix):]: v
        for k, v in ckpt_state.items()
        if k.split(prefix)[-1] in backbone.state_dict()
    }
    missing, unexpected = backbone.load_state_dict(filtered, strict=True)
    print(f"  Weights loaded: {len(filtered)}  |  missing: {len(missing)}  unexpected: {len(unexpected)}")

    backbone.to(device).eval()
    return backbone


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def build_test_df(data_path: str):
    """ChestXray-14/test_list.txt → test DataFrame, y_true (N, 14)."""
    csv_head = [
        'path', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Lung Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia',
    ]
    df = pd.read_csv(os.path.join('ChestXray-14', 'test_list.txt'), sep=' ', names=csv_head)
    key = csv_head[1:]
    df['No Finding'] = (df[key].sum(axis=1) == 0).astype(int)
    df['Image Index'] = df['path'].apply(os.path.basename)
    df.insert(0, 'Image Index', df.pop('Image Index'))

    img_map = {
        os.path.basename(p): p
        for p in glob(os.path.join(data_path, 'images*', '*', '*.png'))
    }
    df['path'] = df['Image Index'].map(img_map)
    df = df.rename(columns={'path': 'Path', 'Lung Mass': 'Mass', 'Lung Nodule': 'Nodule'})

    y_true = df[CLASS_NAMES].values.astype(np.int32)   # (N, 14)
    return df, y_true


# ─────────────────────────────────────────────────────────────
# Inference: logit (N, 28)
# ─────────────────────────────────────────────────────────────

def run_inference(backbone, test_df: pd.DataFrame,
                  device: str, batch_size: int) -> np.ndarray:
    """
    dqn_shot_classification(mcq=True) → avg logit (N, 28).
      [:, :14] = positive,  [:, 14:] = negative
    """
    processed_txt = backbone.process_text(ALL_PROMPTS, device)
    batches = split_list(test_df['Path'].tolist(), batch_size)

    all_logits = []
    with torch.no_grad():
        for paths in tqdm(batches, desc="Inference"):
            imgs = backbone.process_img(paths, device)
            sim = CARZero.dqn_shot_classification(
                backbone, imgs, processed_txt,
                multi=True, mcq=True, torch_tensor=False,
            )  # (B, 28) numpy
            all_logits.append(sim)

    return np.concatenate(all_logits, axis=0)   # (N, 28)


# ─────────────────────────────────────────────────────────────
# ODIN
# ─────────────────────────────────────────────────────────────

def run_odin(backbone, test_df: pd.DataFrame,
             device: str, batch_size: int, eps: float) -> np.ndarray:
    """
    ODIN uncertainty (N, 14).  높을수록 불확실 (= 1 - confidence).

    클래스 k마다 (ODIN/RiskControl.py 패턴):
      1. 전체 28개 prompt로 forward (torch_tensor=True, grad 유지)
      2. chosen_logp 기반 grad w.r.t. 입력 이미지
      3. x_pert = x + eps * sign(grad)
      4. 클래스 k의 2개 prompt로 재추론 → max softmax confidence
    """
    processed_txt = backbone.process_text(ALL_PROMPTS, device)
    batches = split_list(test_df['Path'].tolist(), batch_size)

    all_conf = []

    for paths in tqdm(batches, desc="ODIN"):
        imgs = backbone.process_img(paths, device)  # (B, C, H, W)
        imgs.requires_grad_(True)

        # 전체 prompt forward (grad 추적)
        sim = CARZero.dqn_shot_classification(
            backbone, imgs, processed_txt,
            multi=True, mcq=True, torch_tensor=True,
        )  # (B, 28) tensor

        pos = sim[:, :14]   # (B, 14)
        neg = sim[:, 14:]   # (B, 14)

        conf_k_list = []
        for k in range(14):
            # [neg_k, pos_k] 순서로 log_softmax → max log-prob
            logit_k = torch.stack([neg[:, k], pos[:, k]], dim=-1)   # (B, 2)
            chosen_logp = torch.log_softmax(logit_k, dim=-1).max(dim=-1).values  # (B,)

            grad = torch.autograd.grad(
                outputs=(-chosen_logp).sum(),
                inputs=imgs,
                retain_graph=(k < 13),
                create_graph=False,
            )[0]  # (B, C, H, W)

            x_pert = imgs.detach() + eps * torch.sign(grad.detach())

            # 클래스 k의 [pos_k, neg_k] 2개 prompt로 재추론
            txt_k = backbone.process_text([ALL_PROMPTS[k], ALL_PROMPTS[k + 14]], device)
            with torch.no_grad():
                sim_k = CARZero.dqn_shot_classification(
                    backbone, x_pert, txt_k,
                    multi=True, mcq=True, torch_tensor=False,
                )  # (B, 2) numpy: [pos_k, neg_k]
                prob = torch.softmax(torch.tensor(sim_k, dtype=torch.float32), dim=-1)
                conf_k_list.append(prob.max(dim=-1).values.numpy())  # (B,)

        all_conf.append(np.stack(conf_k_list, axis=1))   # (B, 14)

    odin_conf = np.concatenate(all_conf, axis=0)   # (N, 14)
    return 1.0 - odin_conf                          # uncertainty


# ─────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────

def print_classification(y_true: np.ndarray,
                         p_logit: np.ndarray,
                         n_logit: np.ndarray) -> None:
    """Inference_NIH14.py 스타일: Pos / Neg / PNC 분류 성능 테이블 출력."""
    pd.options.display.float_format = "{:.4f}".format

    print("\n[Positive Results]")
    print(calculate_metric(p_logit, y_true, CLASS_NAMES_15, nf=False).to_string())

    print("\n[Negative Results]")
    print(calculate_metric(n_logit, y_true, CLASS_NAMES_15, nf=False, neg=True).to_string())

    print("\n[PNC Results]")
    pnc_prob = calculate_pnc_logit(p_logit, n_logit)
    print(calculate_metric(pnc_prob, y_true, CLASS_NAMES_15, nf=False).to_string())


# ─────────────────────────────────────────────────────────────
# Uncertainty score functions  (higher = more uncertain)
# ─────────────────────────────────────────────────────────────

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def score_msp(p, n):      return 1.0 - np.maximum(calculate_pnc_logit(p, n), calculate_pnc_logit(n, p))
def score_energy(p, n):   return -np.log(np.exp(p) + np.exp(n))
def score_maxlogit(p, n): return -np.maximum(p, n)
def score_edl(p, n):      return 2.0 / (_softplus(p) + 1.0 + _softplus(n) + 1.0)


# ─────────────────────────────────────────────────────────────
# AURC
# ─────────────────────────────────────────────────────────────

def calculate_aurc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    label_names: List[str],
    coverage_point: float = 0.9,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    N, L = y_true.shape
    u = np.asarray(uncertainty)
    u_mat = np.broadcast_to(u[:, None], (N, L)).copy() if u.ndim == 1 else u
    k_cov = max(1, min(N, int(np.ceil(coverage_point * N))))
    cov_key = f"Risk@{int(coverage_point * 100)}"

    aurcs, rfulls, rcovs, per_label = [], [], [], {}
    for j, name in enumerate(label_names):
        err    = (y_pred[:, j] != y_true[:, j]).astype(np.float64)
        order  = np.argsort(u_mat[:, j])
        cumsum = np.cumsum(err[order])
        k      = np.arange(1, N + 1)
        risk   = cumsum / (k + eps)

        aurc   = float(np.trapz(np.r_[0.0, risk], np.r_[0.0, k / N]))
        r_full = float(risk[-1])
        r_cov  = float(cumsum[k_cov - 1] / (k_cov + eps))

        per_label[name] = {"aurc": aurc, "R(1)": r_full, cov_key: r_cov}
        aurcs.append(aurc); rfulls.append(r_full); rcovs.append(r_cov)

    return {
        "macro": {
            "aurc":  float(np.nanmean(aurcs)),
            cov_key: float(np.nanmean(rcovs)),
            "R(1)":  float(np.nanmean(rfulls)),
        },
        "per_label": per_label,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    VALID_METHODS = ["msp", "energy", "odin", "maxlogit", "edl"]

    parser = argparse.ArgumentParser(
        description="Bi-EDL Inference: classification + uncertainty"
    )
    parser.add_argument("--ckpt_path",  required=True,
                        help="Lightning 체크포인트 (.ckpt)")
    parser.add_argument("--cfg_path",   required=True,
                        help="OmegaConf 설정 파일 (.yaml)")
    parser.add_argument("--data_path",  required=True,
                        help="NIH 데이터셋 루트 경로")
    parser.add_argument("--method",     nargs="+", choices=VALID_METHODS,
                        default=VALID_METHODS, metavar="METHOD",
                        help=f"불확실성 방법 선택 (복수 가능): {VALID_METHODS}")
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--odin_eps",   type=float, default=0.001,
                        help="ODIN perturbation magnitude (default: 0.001)")
    parser.add_argument("--coverage",   type=float, default=0.9,
                        help="Risk@coverage 기준점 (default: 0.9)")
    parser.add_argument("--per_label",  action="store_true",
                        help="클래스별 AURC 상세 출력")
    args = parser.parse_args()

    device   = args.device if torch.cuda.is_available() else "cpu"
    selected = set(args.method)
    print(f"Device: {device}  |  Methods: {args.method}")

    # ── 모델 로드 ───────────────────────────────────────────
    print(f"\nLoading config    : {args.cfg_path}")
    cfg = OmegaConf.load(args.cfg_path)
    print(f"Loading checkpoint: {args.ckpt_path}")
    backbone = load_backbone(args.ckpt_path, cfg, device)

    # ── 데이터 로드 ─────────────────────────────────────────
    print("\nBuilding test DataFrame ...")
    test_df, y_true = build_test_df(args.data_path)
    print(f"  Samples: {len(test_df)}")

    # ── Inference → logit (N, 28) ───────────────────────────
    print("\nRunning inference ...")
    logit   = run_inference(backbone, test_df, device, args.batch_size)
    p_logit = logit[:, :14]    # positive (N, 14)
    n_logit = logit[:, 14:]    # negative (N, 14)
    y_pred  = (p_logit > n_logit).astype(int)

    # ── 분류 성능 ────────────────────────────────────────────
    print_classification(y_true, p_logit, n_logit)

    # ── 불확실성 스코어 ──────────────────────────────────────
    SCORE_FNS = {
        "msp":      ("MSP",      lambda: score_msp(p_logit, n_logit)),
        "energy":   ("Energy",   lambda: score_energy(p_logit, n_logit)),
        "maxlogit": ("MaxLogit", lambda: score_maxlogit(p_logit, n_logit)),
        "edl":      ("EDL",      lambda: score_edl(p_logit, n_logit)),
    }
    methods: Dict[str, np.ndarray] = {}
    for key, (label, fn) in SCORE_FNS.items():
        if key in selected:
            methods[label] = fn()

    if "odin" in selected:
        print(f"\nRunning ODIN (eps={args.odin_eps}) ...")
        methods["ODIN"] = run_odin(backbone, test_df, device, args.batch_size, args.odin_eps)

    # ── AURC 계산 및 출력 ────────────────────────────────────
    cov_key = f"Risk@{int(args.coverage * 100)}"
    results: Dict[str, Dict] = {}
    for name, unc in methods.items():
        results[name] = calculate_aurc(
            y_true, y_pred, unc, CLASS_NAMES, args.coverage
        )

    W = 12
    print(f"\n[Uncertainty Comparison — AURC]")
    print(f"{'=' * (W + 32)}")
    print(f"{'Method':<{W}} {'AURC':>8} {'R(1)':>8} {cov_key:>10}")
    print(f"{'=' * (W + 32)}")
    for name, res in results.items():
        m = res["macro"]
        print(f"{name:<{W}} {m['aurc']:>8.4f} {m['R(1)']:>8.4f} {m[cov_key]:>10.4f}")
    print(f"{'=' * (W + 32)}")

    if args.per_label:
        for name, res in results.items():
            print(f"\n── {name} ──")
            print(f"  {'Class':<26} {'AURC':>8} {'R(1)':>8} {cov_key:>10}")
            for cls, vals in res["per_label"].items():
                print(f"  {cls:<26} {vals['aurc']:>8.4f} {vals['R(1)']:>8.4f} {vals[cov_key]:>10.4f}")

if __name__ == "__main__":
    main()
