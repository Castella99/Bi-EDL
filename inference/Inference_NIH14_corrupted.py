import sys
import os
os.chdir('/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero')
sys.path.append('/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero')

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
from finetuning_inference import obtain_attn, obtain_logit, obtain_ood_logit, obtain_simr, calculate_pnc_logit, calculate_metric
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

    
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

def sev_idx(severity: int) -> int:
    if not (1 <= severity <= 5):
        raise ValueError("severity는 1~5 범위여야 합니다.")
    return severity - 1

def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0

def _np_to_pil(x: np.ndarray) -> Image.Image:
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(x)

def pil_pixelate(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    w2 = max(1, int(round(w * scale)))
    h2 = max(1, int(round(h * scale)))
    small = F.resize(img, (h2, w2), interpolation=InterpolationMode.NEAREST)
    return F.resize(small, (h, w), interpolation=InterpolationMode.NEAREST)

def pil_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    x = _pil_to_np(img)
    x = x + sigma * np.random.randn(*x.shape)
    return _np_to_pil(np.clip(x, 0.0, 1.0))

def pil_speckle_noise(img: Image.Image, sigma: float) -> Image.Image:
    x = _pil_to_np(img)
    x = x + x * sigma * np.random.randn(*x.shape)
    return _np_to_pil(np.clip(x, 0.0, 1.0))

def pil_impulse_noise(img: Image.Image, amount: float) -> Image.Image:
    x = _pil_to_np(img)
    # (주의) img가 RGB면 shape=(H,W,3)이라 아래는 그대로면 오류입니다.
    # ChestMNIST가 grayscale이라는 전제라면 H,W가 맞습니다.
    h, w = x.shape
    mask = np.random.rand(h, w) < amount
    salt = np.random.rand(h, w) < 0.5
    x[mask & salt] = 1.0
    x[mask & (~salt)] = 0.0
    return _np_to_pil(x)

def pil_shot_noise(img: Image.Image, lam: float) -> Image.Image:
    x = _pil_to_np(img)
    y = np.random.poisson(x * lam) / lam
    return _np_to_pil(np.clip(y, 0.0, 1.0))

def pil_gaussian_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return T.GaussianBlur(kernel_size=kernel_size)(img)

def pil_brightness(img: Image.Image, factor: float) -> Image.Image:
    return F.adjust_brightness(img, factor)

def pil_contrast(img: Image.Image, factor: float) -> Image.Image:
    return F.adjust_contrast(img, factor)

def pil_gamma(img: Image.Image, gamma: float) -> Image.Image:
    return F.adjust_gamma(img, gamma=gamma, gain=1.0)

# --------------------------
# 추가: random rotation / crop / affine
# --------------------------

def pil_random_rotation(img: Image.Image, max_degrees: float) -> Image.Image:
    # [-max_degrees, +max_degrees] 균일 샘플
    angle = random.uniform(-max_degrees, max_degrees)
    # ChestMNIST 흑백 가정: fill=0
    return F.rotate(
        img,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=0,
        center=None,
    )

def pil_random_crop_resize(img: Image.Image, keep_ratio: float) -> Image.Image:
    # keep_ratio: 0~1 (원본에서 남길 비율)
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio는 (0, 1] 범위여야 합니다.")

    w, h = img.size
    tw = max(1, int(round(w * keep_ratio)))
    th = max(1, int(round(h * keep_ratio)))

    if tw == w and th == h:
        return img

    left = 0 if w == tw else random.randint(0, w - tw)
    top = 0 if h == th else random.randint(0, h - th)

    cropped = F.crop(img, top=top, left=left, height=th, width=tw)
    # 원래 크기로 복원 (정보 손실 corruption)
    return F.resize(cropped, (h, w), interpolation=InterpolationMode.BILINEAR)

def pil_random_affine(
    img: Image.Image,
    degrees: float,
    translate: Tuple[float, float],
    scale: Tuple[float, float],
    shear: Tuple[float, float],
) -> Image.Image:
    # torchvision의 RandomAffine 샘플링 규칙을 단순화해 직접 샘플
    angle = random.uniform(-degrees, degrees)

    max_dx = translate[0] * img.size[0]
    max_dy = translate[1] * img.size[1]
    tx = random.uniform(-max_dx, max_dx)
    ty = random.uniform(-max_dy, max_dy)

    sc = random.uniform(scale[0], scale[1])

    shx = random.uniform(-shear[0], shear[0])
    shy = random.uniform(-shear[1], shear[1])

    return F.affine(
        img,
        angle=angle,
        translate=[tx, ty],
        scale=sc,
        shear=[shx, shy],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
        center=None,
    )

class ChestMNISTCorruptPIL:
    """
    MedMNIST-C (ChestMNIST) strict corruption
    - PIL domain
    - single corruption per sample
    - severity 1–5
    - JPEG 제외
    """

    def __init__(
        self,
        registry: Dict[str, List],
        p_identity: float = 0.0,
        seed: Optional[int] = None,
        gaussian_blur_use_first5: bool = True,
    ):
        self.r = registry
        self.keys = list(registry.keys())
        self.p_identity = p_identity
        self.gaussian_blur_use_first5 = gaussian_blur_use_first5

        if seed is not None:
            random.seed(seed)

    def __call__(
        self,
        img: Image.Image,
        severity: Optional[int] = None,
        corruption: Optional[str] = None,
    ) -> Image.Image:

        if random.random() < self.p_identity:
            return img

        sev = severity if severity is not None else random.randint(1, 5)
        corr = corruption if corruption is not None else random.choice(self.keys)
        i = sev_idx(sev)

        if corr == "pixelate":
            return pil_pixelate(img, self.r["pixelate"][i])

        if corr == "gaussian_noise":
            return pil_gaussian_noise(img, self.r["gaussian_noise"][i])

        if corr == "speckle_noise":
            return pil_speckle_noise(img, self.r["speckle_noise"][i])

        if corr == "impulse_noise":
            return pil_impulse_noise(img, self.r["impulse_noise"][i])

        if corr == "shot_noise":
            return pil_shot_noise(img, self.r["shot_noise"][i])

        if corr == "gaussian_blur":
            ks = self.r["gaussian_blur"]
            if self.gaussian_blur_use_first5:
                ks = ks[:5]
            return pil_gaussian_blur(img, ks[i])

        if corr == "brightness_up":
            return pil_brightness(img, self.r["brightness_up"][i])

        if corr == "brightness_down":
            return pil_brightness(img, self.r["brightness_down"][i])

        if corr == "contrast_up":
            return pil_contrast(img, self.r["contrast_up"][i])

        if corr == "contrast_down":
            return pil_contrast(img, self.r["contrast_down"][i])

        if corr == "gamma_corr_up":
            return pil_gamma(img, self.r["gamma_corr_up"][i])

        if corr == "gamma_corr_down":
            return pil_gamma(img, self.r["gamma_corr_down"][i])

        if corr == "random_crop":
            return pil_random_crop_resize(img, self.r["random_crop"][i])

        if corr == "random_affine":
            params = self.r["random_affine"][i]
            # params: (degrees, (tx,ty), (smin,smax), (shx,shy))
            degrees, translate, scale, shear = params
            return pil_random_affine(
                img,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
            )

        raise KeyError(f"Unknown corruption: {corr}")

CHESTMNIST_REGISTRY = {
    "pixelate": [0.30, 0.25, 0.20, 0.15, 0.10],
    "gaussian_noise": [0.04, 0.08, 0.12, 0.18, 0.26],
    "speckle_noise": [0.05, 0.15, 0.20, 0.35, 0.45],
    "impulse_noise": [0.01, 0.03, 0.06, 0.09, 0.17],
    "shot_noise": [60, 25, 18, 10, 5],
    "gaussian_blur": [3, 5, 7, 9, 11, 13],
    "brightness_up": [1.1, 1.2, 1.3, 1.4, 1.5],
    "brightness_down": [0.9, 0.8, 0.7, 0.6, 0.5],
    "contrast_up": [1.1, 1.2, 1.3, 1.4, 1.6],
    "contrast_down": [0.9, 0.8, 0.7, 0.6, 0.4],
    "gamma_corr_up": [1.1, 1.2, 1.3, 1.4, 1.6],
    "gamma_corr_down": [0.9, 0.8, 0.7, 0.6, 0.4],
    "random_crop": [0.95, 0.90, 0.85, 0.80, 0.75],
    "random_affine": [
        (5,  (0.02, 0.02), (0.98, 1.02), (2, 2)),
        (10, (0.04, 0.04), (0.96, 1.04), (4, 4)),
        (15, (0.06, 0.06), (0.94, 1.06), (6, 6)),
        (20, (0.08, 0.08), (0.92, 1.08), (8, 8)),
        (25, (0.10, 0.10), (0.90, 1.10), (10, 10)),
    ],
}

pd.options.display.float_format = '{:.3f}'.format

plt.style.use('default')

class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']

positive = {"0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Pleural Effusion"], "3": ["There is Pulmonary Infiltration"], "4": ["There is Pulmonary Mass"], "5": ["There is Lung Nodule"], "6": ["There is Pneumonia"], "7": ["There is Pneumothorax"], "8": ["There is Pulmonary Consolidation"], "9": ["There is Pulmonary Edema"], "10": ["There is Pulmonary Emphysema"], "11": ["There is Fibrosis"], "12": ["There is Pleural Thickening"], "13": ["There is Hernia"], "14" : ["There is no Finding"]}
negative = {"0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Pleural Effusion"], "3": ["There is no Pulmonary Infiltration"], "4": ["There is no Pulmonary Mass"], "5": ["There is no Lung Nodule"], "6": ["There is no Pneumonia"], "7": ["There is no Pneumothorax"], "8": ["There is no Pulmonary Consolidation"], "9": ["There is no Pulmonary Edema"], "10": ["There is no Pulmonary Emphysema"], "11": ["There is no Fibrosis"], "12": ["There is no Pleural Thickening"], "13": ["There is no Hernia"]}

prompts = ["There is Atelectasis", "There is Cardiomegaly", "There is Pleural Effusion", "There is Pulmonary Infiltration", "There is Pulmonary Mass", "There is Lung Nodule", "There is Pneumonia", "There is Pneumothorax", "There is Pulmonary Consolidation", "There is Pulmonary Edema", "There is Pulmonary Emphysema", "There is Fibrosis", "There is Pleural Thickening", "There is Hernia", "There is no Finding", "There is no Atelectasis", "There is no Cardiomegaly", "There is no Pleural Effusion", "There is no Pulmonary Infiltration", "There is no Pulmonary Mass", "There is no Lung Nodule", "There is no Pneumonia", "There is no Pneumothorax", "There is no Pulmonary Consolidation", "There is no Pulmonary Edema", "There is no Pulmonary Emphysema", "There is no Fibrosis", "There is no Pleural Thickening", "There is no Hernia"]

def main(args) :
    cfg = OmegaConf.load(os.path.join(args.result_dir, 'config.yaml')) if args.result_dir else None
    if not args.result_dir :
        args.result_dir = "./logs/CARZero_Zeroshot/"
        cfg = OmegaConf.load(os.path.join(args.result_dir, 'config.yaml'))
        CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device="cuda", multi=False, cfg=cfg)
    
    else :
        checkpoint =  os.path.join(args.result_dir, "checkpoints/best_model.ckpt")
        CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device="cuda", multi=cfg.model.CARZero.multi, cfg=cfg)
        ckpt_state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
        fixed_ckpt_dict = {k.split("CARZero_model.")[-1]: v for k, v in ckpt_state_dict.items() if k.split("CARZero_model.")[-1] in CARZero_model.state_dict()}
        CARZero_model.load_state_dict(fixed_ckpt_dict, strict=True)
    CARZero_model.eval()
    
    print(args.result_dir)
    
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

    pil_corrupt = ChestMNISTCorruptPIL(
        registry=CHESTMNIST_REGISTRY,
        p_identity=0.0,
    )


    # df_train = df[~df['Image Index'].isin(df_test['Image Index'])].reset_index(drop=True)
    # df_train = df_train.drop(columns=['Finding Labels'], errors='ignore')
    
    # train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=42)
    #test_df = test_df.iloc[:256]
    
    # val_p_logit, val_p_i2t_attn, val_p_t2i_attn = obtain_attn(val_df, positive, CARZero_model, device="cuda:0")
    # val_n_logit, val_n_i2t_attn, val_n_t2i_attn = obtain_attn(val_df, negative, CARZero_model, device="cuda:0")
    
    # p_logit_param = learn_temperature(val_p_logit[:,:-1], val_df[class_name[:-1]].values.argmax(axis=1), device="cuda:0")
    # n_logit_param = learn_temperature(val_n_logit[:,:-1], val_df[class_name[:-1]].values.argmax(axis=1), device="cuda:0")
    
    # p_logit, p_i2t_attn, p_t2i_attn = obtain_attn(test_df, positive, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    # n_logit, n_i2t_attn, n_t2i_attn = obtain_attn(test_df, negative, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    logit = obtain_ood_logit(test_df, prompts, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi, corrupt=pil_corrupt, severity=5)
    
    np.save(os.path.join(args.result_dir, "corrupted_logit.npy"), logit)
    p_logit = logit[:, :len(class_name)]
    n_logit = logit[:, len(class_name)-1:]
    
    pnc_logit = calculate_pnc_logit(p_logit[:,:-1], n_logit)
    
    pos_results = calculate_metric(p_logit, true_labels, class_name, nf=True)
    neg_results = calculate_metric(n_logit, true_labels, class_name, neg=True)
    pnc_results = calculate_metric(pnc_logit, true_labels, class_name)
    
    print("Positive Results:")
    print(pos_results)
    print("Negative Results:")
    print(neg_results)
    print("Positive-Negative Combined Results:")
    print(pnc_results)
    
    save_dir = os.path.join(args.result_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    pos_results.to_csv(os.path.join(save_dir, "NIH_pos_results.csv"))
    neg_results.to_csv(os.path.join(save_dir, "NIH_neg_results.csv"))
    pnc_results.to_csv(os.path.join(save_dir, "NIH_pnc_results.csv"))
    
    if args.dir :
        p_logit_i2t, p_i2t_feat, p_t2i_feat = obtain_simr(test_df, positive, CARZero_model, device="cuda:0", mode='i2t', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        n_logit_i2t, n_i2t_feat, n_t2i_feat = obtain_simr(test_df, negative, CARZero_model, device="cuda:0", mode='i2t', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        
        p_logit_t2i, p_i2t_feat, p_t2i_feat = obtain_simr(test_df, positive, CARZero_model, device="cuda:0", mode='t2i', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        n_logit_t2i, n_i2t_feat, n_t2i_feat = obtain_simr(test_df, negative, CARZero_model, device="cuda:0", mode='t2i', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        
        pnc_i2t_logit = calculate_pnc_logit(p_logit_i2t[:, :-1], n_logit_i2t)
        pnc_t2i_logit = calculate_pnc_logit(p_logit_t2i[:, :-1], n_logit_t2i)
        
        pos_i2t_results = calculate_metric(p_logit_i2t, true_labels, class_name, nf=True)
        neg_i2t_results = calculate_metric(n_logit_i2t, true_labels, class_name, neg=True)
        pnc_i2t_results = calculate_metric(pnc_i2t_logit, true_labels, class_name)
        
        pos_t2i_results = calculate_metric(p_logit_t2i, true_labels, class_name, nf=True)
        neg_t2i_results = calculate_metric(n_logit_t2i, true_labels, class_name, neg=True)
        pnc_t2i_results = calculate_metric(pnc_t2i_logit, true_labels, class_name)
        
        print("Positive I2T Results:")
        print(pos_i2t_results)
        print("Negative I2T Results:")
        print(neg_i2t_results)
        print("Positive-Negative Combined I2T Results:")
        print(pnc_i2t_results)
        
        print("Positive T2I Results:")
        print(pos_t2i_results)
        print("Negative T2I Results:")
        print(neg_t2i_results)
        print("Positive-Negative Combined T2I Results:")
        print(pnc_t2i_results)
        
        pos_i2t_results.to_csv(os.path.join(save_dir, "NIH_pos_i2t_results.csv"))
        neg_i2t_results.to_csv(os.path.join(save_dir, "NIH_neg_i2t_results.csv"))
        pnc_i2t_results.to_csv(os.path.join(save_dir, "NIH_pnc_i2t_results.csv"))
        
        pos_t2i_results.to_csv(os.path.join(save_dir, "NIH_pos_t2i_results.csv"))
        neg_t2i_results.to_csv(os.path.join(save_dir, "NIH_neg_t2i_results.csv"))
        pnc_t2i_results.to_csv(os.path.join(save_dir, "NIH_pnc_t2i_results.csv")) 
        
    if args.tsne : 
        pn_i2t_features_tsne = [TSNE(n_components=2, random_state=42).fit_transform(np.concatenate((p_i2t_feat[:,i,:], n_i2t_feat[:,i,:]), axis=0)) for i in tqdm(range(n_logit_i2t.shape[1]))]
        pn_t2i_features_tsne = [TSNE(n_components=2, random_state=42).fit_transform(np.concatenate((p_t2i_feat[:,i,:], n_t2i_feat[:,i,:]), axis=0)) for i in tqdm(range(n_logit_t2i.shape[1]))]
        
        pn_i2t_features_tsne = np.array(pn_i2t_features_tsne)
        pn_t2i_features_tsne = np.array(pn_t2i_features_tsne)
        
        p_i2t_feat_tsne = pn_i2t_features_tsne[:, :p_i2t_feat.shape[0], :]
        p_t2i_feat_tsne = pn_t2i_features_tsne[:, :p_t2i_feat.shape[0], :]
        n_i2t_feat_tsne = pn_i2t_features_tsne[:, p_i2t_feat.shape[0]:, :]
        n_t2i_feat_tsne = pn_t2i_features_tsne[:, p_t2i_feat.shape[0]:, :]
        
        plt.figure(figsize=(20, 20))
        for i in range(n_logit_i2t.shape[1]):
            true_sample = (test_df.iloc[:,i+3]==1)
            false_sample = (test_df.iloc[:,i+3]==0)
            plt.subplot(4, 4, i+1)
            plt.scatter(p_i2t_feat_tsne[i][false_sample, 0], p_i2t_feat_tsne[i][false_sample, 1], s=0.5, c='red', label='Positive - False', alpha=0.5)
            plt.scatter(n_i2t_feat_tsne[i][false_sample, 0], n_i2t_feat_tsne[i][false_sample, 1], s=0.5, c='green', label='Negative - False', alpha=0.5)
            plt.scatter(p_i2t_feat_tsne[i][true_sample, 0], p_i2t_feat_tsne[i][true_sample, 1], s=1, c='blue', label='Positive - True', alpha=0.5)
            plt.scatter(n_i2t_feat_tsne[i][true_sample, 0], n_i2t_feat_tsne[i][true_sample, 1], s=1, c='orange', label='Negative - True', alpha=0.5)
            plt.xlabel('t-SNE Component 1')
            plt.title(f"Class - {class_name[i]}, PosAUC - {pos_i2t_results.loc[class_name[i], 'auc']:.3f}, NegAUC - {neg_i2t_results.loc[class_name[i], 'auc']:.3f} ")
            plt.legend()
            plt.axis('off')   
        plt.tight_layout()
        plt.savefig(f"{save_dir}/NIH_i2t_tsne_plot.png")
        plt.close()
        
        plt.figure(figsize=(20, 20))
        for i in range(n_logit_t2i.shape[1]):
            true_sample = (test_df.iloc[:,i+3]==1)
            false_sample = (test_df.iloc[:,i+3]==0)
            plt.subplot(4, 4, i+1)
            plt.scatter(p_t2i_feat_tsne[i][false_sample, 0], p_t2i_feat_tsne[i][false_sample, 1], s=0.5, c='red', label='Positive - False', alpha=0.5)
            plt.scatter(n_t2i_feat_tsne[i][false_sample, 0], n_t2i_feat_tsne[i][false_sample, 1], s=0.5, c='green', label='Negative - False', alpha=0.5)
            plt.scatter(p_t2i_feat_tsne[i][true_sample, 0], p_t2i_feat_tsne[i][true_sample, 1], s=1, c='blue', label='Positive - True', alpha=0.5)
            plt.scatter(n_t2i_feat_tsne[i][true_sample, 0], n_t2i_feat_tsne[i][true_sample, 1], s=1, c='orange', label='Negative - True', alpha=0.5)
            plt.xlabel('t-SNE Component 1')
            plt.title(f"Class - {class_name[i]}, PosAUC - {pos_t2i_results.loc[class_name[i], 'auc']:.3f}, NegAUC - {neg_t2i_results.loc[class_name[i], 'auc']:.3f} ")
            plt.legend()
            plt.axis('off')   
        plt.tight_layout()
        plt.savefig(f"{save_dir}/NIH_t2i_tsne_plot.png")
        plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--tsne", type=str, default=False)
    parser.add_argument("--dir", type=str, default=False)
    args = parser.parse_args()
    
    main(args)