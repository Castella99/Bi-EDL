# %%
import os
os.chdir("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero")
import sys
sys.path.append("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero")

# %%
import torch
import CARZero
import pandas as pd 
import json
import numpy as np
from utils import *
from sklearn.preprocessing import MultiLabelBinarizer
from glob import glob
from tqdm import tqdm
import CARZero.builder as builder
from omegaconf import OmegaConf
import copy
from uncertainty_utils import *
from finetuning_inference import obtain_attn, obtain_simr, calculate_pnc_logit, calculate_metric
import argparse
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:.3f}'.format

plt.style.use('default')

# %%
class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']

positive = {"0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Pleural Effusion"], "3": ["There is Pulmonary Infiltration"], "4": ["There is Pulmonary Mass"], "5": ["There is Lung Nodule"], "6": ["There is Pneumonia"], "7": ["There is Pneumothorax"], "8": ["There is Pulmonary Consolidation"], "9": ["There is Pulmonary Edema"], "10": ["There is Pulmonary Emphysema"], "11": ["There is Fibrosis"], "12": ["There is Pleural Thickening"], "13": ["There is Hernia"], "14" : ["There is no Finding"]}
negative = {"0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Pleural Effusion"], "3": ["There is no Pulmonary Infiltration"], "4": ["There is no Pulmonary Mass"], "5": ["There is no Lung Nodule"], "6": ["There is no Pneumonia"], "7": ["There is no Pneumothorax"], "8": ["There is no Pulmonary Consolidation"], "9": ["There is no Pulmonary Edema"], "10": ["There is no Pulmonary Emphysema"], "11": ["There is no Fibrosis"], "12": ["There is no Pleural Thickening"], "13": ["There is no Hernia"]}

prompts = ["There is Atelectasis", "There is Cardiomegaly", "There is Pleural Effusion", "There is Pulmonary Infiltration", "There is Pulmonary Mass", "There is Lung Nodule", "There is Pneumonia", "There is Pneumothorax", "There is Pulmonary Consolidation", "There is Pulmonary Edema", "There is Pulmonary Emphysema", "There is Fibrosis", "There is Pleural Thickening", "There is Hernia", "There is no Finding", "There is no Atelectasis", "There is no Cardiomegaly", "There is no Pleural Effusion", "There is no Pulmonary Infiltration", "There is no Pulmonary Mass", "There is no Lung Nodule", "There is no Pneumonia", "There is no Pneumothorax", "There is no Pulmonary Consolidation", "There is no Pulmonary Edema", "There is no Pulmonary Emphysema", "There is no Fibrosis", "There is no Pleural Thickening", "There is no Hernia"]

# %%
checkpoint = "/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero/logs/CARZero_Finetuning_MCQ_Only_ver2/CARZero_MCQ_single_direction_ver2/checkpoints/best_model.ckpt"

# %%
cfg = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'config.yaml'))

# %%
CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device="cuda", multi=cfg.model.CARZero.multi, cfg=cfg)
ckpt_state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
fixed_ckpt_dict = {k.split("CARZero_model.")[-1]: v for k, v in ckpt_state_dict.items() if k.split("CARZero_model.")[-1] in CARZero_model.state_dict()}
CARZero_model.load_state_dict(fixed_ckpt_dict, strict=False)
CARZero_model.eval()

# %%
data_path = '/shared/home/mai/Taehun/Uncertainty/data/NIH'
with open(os.path.join(data_path, 'test_list.txt'), 'r') as f :
    test_list = f.readlines()
test_list = [x.strip() for x in test_list]
path = os.path.join(data_path, 'Data_Entry_2017.csv')
df = pd.read_csv(path)
df['Finding Labels'] = df['Finding Labels'].str.replace('_', ' ', regex=False)
df = df[['Image Index', 'Finding Labels']]
img_path = {os.path.basename(x): x for x in glob(os.path.join(data_path, 'images*', '*', '*.png'))}
df['Path'] = df['Image Index'].map(img_path)
for name in class_name:
    df[name] = df['Finding Labels'].apply(lambda x: 1 if name in x else 0)
    
label_path = 'ChestXray-14'
data_path = '/shared/home/mai/Taehun/Uncertainty/data/NIH'
csv_head = ['path', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Lung Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']
label_file_path = os.path.join(label_path, 'test_list.txt')
df_test = pd.read_csv(label_file_path, sep=' ', names=csv_head)
key = csv_head[1:]
label = df_test[key].values
# add 'No Finding' column: 1 if all other label columns are 0, else 0
df_test['No Finding'] = (df_test[key].sum(axis=1) == 0).astype(int)

# %%
# update key and label to include the new column
key = key + ['No Finding']
label = df_test[key].values
df_test['Image Index'] = df_test['path'].apply(lambda x: os.path.basename(x))
if 'Image Index' in df_test.columns:
    df_test.insert(0, 'Image Index', df_test.pop('Image Index'))
img_path = {os.path.basename(x): x for x in glob(os.path.join(data_path, 'images*', '*', '*.png'))}
df_test['path'] = df_test['Image Index'].map(img_path)
rename_map = {'path': 'Path', 'Lung Mass': 'Mass', 'Lung Nodule': 'Nodule'}
test_df = df_test.rename(columns=rename_map)

true_labels = test_df.iloc[:, 2:].values

# %%
def obtain_attn_odin(df, texts, CARZero_model, device, multi=True, mcq=True, eps=0.001): 
    # process input images and class prompts 
    ## batchsize
    bs = 32
    image_list = split_list(df['Path'].tolist(), bs)
    processed_txt = CARZero_model.process_text(texts, device)
    
    sims = []
    
    for i, img in tqdm(enumerate(image_list), total=len(image_list), desc="Processing images"):
        processed_imgs = CARZero_model.process_img(img, device)
        processed_imgs = processed_imgs.requires_grad_(True)
        # zero-shot classification on 1000 images
        similarities = CARZero.dqn_shot_classification(
            CARZero_model, processed_imgs, processed_txt, multi=multi, mcq=mcq, torch_tensor=True)
        pos_sim = similarities[:, :14]
        neg_sim = similarities[:, 15:]
        
        sim = torch.stack([pos_sim, neg_sim], dim=-1)
        logp = torch.log_softmax(sim, dim=-1)
        # keep only the maximum log-prob across the last dimension
        y_hat = logp.argmax(dim=-1)

        chosen_logp = logp.gather(dim=-1, index=y_hat.unsqueeze(-1)).squeeze(-1)

        B, K = chosen_logp.shape
        C, H, W = processed_imgs.shape[1:]

        perturbated_imgs = torch.empty((B, K, C, H, W), device=processed_imgs.device, dtype=processed_imgs.dtype)

        # 그래프를 재사용해야 하므로 retain_graph=True 필요
        CARZero_model.zero_grad(set_to_none=True)
        sim_k = []
        for k in range(K):
            loss_k = -chosen_logp[:, k].sum()

            grad_k = torch.autograd.grad(
                outputs=loss_k,
                inputs=processed_imgs,
                retain_graph=True,
                create_graph=False,
                allow_unused=False
            )[0]  # (B,C,H,W)

            x_pert_k = processed_imgs.detach() + eps * torch.sign(grad_k.detach())
            perturbated_imgs[:, k] = x_pert_k
            
            processed_txt_k = CARZero_model.process_text(np.asarray(prompts)[[k, k+15]], device)
            
            perturbated_sim = CARZero.dqn_shot_classification(
                CARZero_model, perturbated_imgs[:, k], processed_txt_k, multi=multi, mcq=mcq, torch_tensor=False)
            perturbated_sim = torch.softmax(torch.tensor(perturbated_sim), dim=-1).numpy()
            perturbated_sim = perturbated_sim.max(axis=-1)
            sim_k.append(perturbated_sim)
        sims.append(np.stack(sim_k, axis=1))

    sims = np.concatenate(sims, axis=0)
    
    return sims

# %%
odin_id = obtain_attn_odin(
    test_df, prompts, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)

# %%
logit = np.load("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero/MICCAI2026/logit.npy")

# %%
np.save("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero/MICCAI2026/ODIN/odin_id.npy", odin_id)

# %%
p_logit = logit[:, :14]
n_logit = logit[:, 15:]

# %%
sigmoid_output = torch.sigmoid(torch.tensor(p_logit)).numpy()
auroc_dict = {}
for i, class_label in enumerate(tqdm(class_name[:-1])):
    true_label = true_labels[:,i]
    
    precision, recall, thresholds = precision_recall_curve(true_label, sigmoid_output[:, i])
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    predicted_labels = (sigmoid_output[:, i] > max_f1_thresh).astype(int)  # 예측값 (threshold 사용)
    
    failure_case = true_label != predicted_labels
    
    failure_detection = roc_auc_score(failure_case.astype(int), odin_id[:, i])
    print(f"{class_label} - Failure Detection AUC: {failure_detection:.4f}")
    auroc_dict[class_label] = failure_detection

# %%
np.mean(list(auroc_dict.values()))

# %%
from typing import Dict, Any, Optional, List
import numpy as np

def multilabel_aurc_and_risk_reduction(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    uncertainty: np.ndarray,
    label_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    report_coverages: Optional[List[float]] = None,
    eps: float = 1e-12,
    return_macro: bool = True,
) -> Dict[str, Any]:
    """
    멀티라벨(질환별) AURC와 Risk Reduction만 계산하여 반환합니다.

    Risk Reduction@α = 1 - Risk@Coverage(α) / Risk@100%

    Inputs
    - y_true: (N, L) binary {0,1}
    - y_pred_or_prob: (N, L) binary {0,1} or probabilities in [0,1]
    - uncertainty: (N,) or (N, L), higher = more uncertain (low uncertainty accepted first)
    - label_names: length L (optional)
    - threshold: used if y_pred_or_prob are probabilities
    - report_coverages: list like [0.8, 0.9, 0.95]
    - return_macro: if True, also return macro averages

    Returns
    {
      "per_label": {
         label_name: {
            "aurc": float,
            "risk_reduction": {alpha: float, ...}
         }, ...
      },
      (optional) "macro": {
         "aurc": float,
         "risk_reduction": {alpha: float, ...}
      }
    }
    """
    y_true = np.asarray(y_true)
    y_pred_or_prob = np.asarray(y_pred_or_prob)

    if y_true.ndim != 2:
        raise ValueError(f"y_true must be 2D (N,L), got {y_true.shape}")
    if y_pred_or_prob.ndim != 2:
        raise ValueError(f"y_pred_or_prob must be 2D (N,L), got {y_pred_or_prob.shape}")

    N, L = y_true.shape
    if y_pred_or_prob.shape != (N, L):
        raise ValueError(f"y_pred_or_prob shape must match y_true {(N,L)}, got {y_pred_or_prob.shape}")

    # 기본 coverage 포인트
    if report_coverages is None:
        report_coverages = [0.8, 0.9, 0.95]
    report_coverages = [float(a) for a in report_coverages]
    for a in report_coverages:
        if not (0.0 < a <= 1.0):
            raise ValueError(f"coverage must be in (0,1], got {a}")

    # label names
    if label_names is None:
        label_names = [f"label_{i}" for i in range(L)]
    if len(label_names) != L:
        raise ValueError(f"label_names length must be L={L}, got {len(label_names)}")

    # uncertainty matrix로 정규화
    u = np.asarray(uncertainty)
    if u.ndim == 1:
        if u.shape[0] != N:
            raise ValueError(f"uncertainty (N,) must have length N={N}, got {u.shape}")
        u_mat = np.repeat(u.reshape(N, 1), L, axis=1)
    elif u.ndim == 2:
        if u.shape != (N, L):
            raise ValueError(f"uncertainty (N,L) must match y_true {(N,L)}, got {u.shape}")
        u_mat = u
    else:
        raise ValueError(f"uncertainty must be (N,) or (N,L), got {u.shape}")

    # y_pred를 binary로 통일
    y_pred = y_pred_or_prob
    uniq = np.unique(y_pred)
    is_binary = (uniq.size <= 2) and np.all(np.isin(uniq, [0, 1]))
    if is_binary:
        y_pred_bin = y_pred.astype(np.int64)
    else:
        if np.any((y_pred < 0) | (y_pred > 1)):
            raise ValueError("y_pred_or_prob looks non-binary and not in [0,1].")
        y_pred_bin = (y_pred >= threshold).astype(np.int64)

    per_label: Dict[str, Any] = {}
    aurc_list = []
    rr_collect = {a: [] for a in report_coverages}

    # 클래스별 계산
    for j, name in enumerate(label_names):
        yt = y_true[:, j].astype(np.int64).reshape(-1)
        yp = y_pred_bin[:, j].astype(np.int64).reshape(-1)
        uj = u_mat[:, j].reshape(-1)

        if yt.shape[0] < 2:
            raise ValueError("Need at least 2 samples per label to form a curve.")

        err = (yp != yt).astype(np.float64)

        # uncertainty 오름차순: confident first
        order = np.argsort(uj, kind="mergesort")
        err_sorted = err[order]

        # RC curve risk(k) = cumsum_err / k, coverage(k)=k/N
        cumsum_err = np.cumsum(err_sorted)
        k = np.arange(1, N + 1)
        risk = cumsum_err / (k + eps)
        coverage = k / N

        # AURC (0,0) anchor 포함
        cov_aug = np.concatenate([[0.0], coverage])
        risk_aug = np.concatenate([[0.0], risk])
        aurc = float(np.trapz(risk_aug, cov_aug))

        # Risk@100% (전체 오류율)
        risk_full = float(risk[-1])

        # Risk reduction at specified coverages
        rr_dict: Dict[float, float] = {}
        for a in report_coverages:
            kk = int(np.ceil(a * N))
            kk = max(1, min(N, kk))
            risk_a = float(cumsum_err[kk - 1] / (kk + eps))

            if risk_full <= eps:
                # 전체 오류가 0이면 정규화가 무의미하므로 0으로 처리
                rr = 0.0
            else:
                rr = float(1.0 - (risk_a / (risk_full + eps)))

            rr_dict[a] = rr
            rr_collect[a].append(rr)

        per_label[name] = {
            "aurc": aurc,
            "risk_reduction": rr_dict
        }

        aurc_list.append(aurc)

    out: Dict[str, Any] = {"per_label": per_label}

    if return_macro:
        macro_aurc = float(np.nanmean(np.array(aurc_list)))
        macro_rr = {a: float(np.nanmean(np.array(vals))) for a, vals in rr_collect.items()}
        out["macro"] = {"aurc": macro_aurc, "risk_reduction": macro_rr}

    return out


# %%
res = multilabel_aurc_and_risk_reduction(
    y_true=true_labels[:, :-1],
    y_pred_or_prob=sigmoid_output,
    uncertainty=odin_id,
    label_names=class_name[:-1],
    threshold=0.5,
    report_coverages=[0.8, 0.9],
)
print(res)