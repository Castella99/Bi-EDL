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
from CARZero.finetuning_inference import obtain_attn, obtain_simr, calculate_pnc_logit, calculate_metric
import argparse
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.style.use('default')

class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']

positive = {"0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Pleural Effusion"], "3": ["There is Pulmonary Infiltration"], "4": ["There is Pulmonary Mass"], "5": ["There is Lung Nodule"], "6": ["There is Pneumonia"], "7": ["There is Pneumothorax"], "8": ["There is Pulmonary Consolidation"], "9": ["There is Pulmonary Edema"], "10": ["There is Pulmonary Emphysema"], "11": ["There is Fibrosis"], "12": ["There is Pleural Thickening"], "13": ["There is Hernia"], "14" : ["There is no Finding"]}

def main(args) :
    cfg = OmegaConf.load(os.path.join(args.result_dir, 'config.yaml')) if args.result_dir else None
    if not args.result_dir:
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
    
    uncertainty_df = pd.read_csv("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/RoentGen/RoentGen_NIH14_uncertainty_results.csv")
    
    uncertaint_median = uncertainty_df['U'].median()
    mad_u = np.median(np.abs(uncertainty_df['U'] - uncertaint_median))
    uncertainty_df['is_high_uncertainty'] = (uncertainty_df['U'] - uncertaint_median) / mad_u > 4.0 
    
    high_uncertainty_df = uncertainty_df[uncertainty_df['is_high_uncertainty'] == True]
    
    high_unc_logit, _, _ = obtain_attn(high_uncertainty_df, positive, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    high_unc_results = calculate_metric(high_unc_logit, high_uncertainty_df.iloc[:, 6:-1].values, class_name, nf=True)
    print("High Uncertainty Results:")
    print(high_unc_results)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--tsne", type=str, default=False)
    args = parser.parse_args()
    
    main(args)