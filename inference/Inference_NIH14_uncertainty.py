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
    
    uncertainty_df['U_quartile'] = pd.qcut(uncertainty_df['U'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    q1_df = uncertainty_df[uncertainty_df['U_quartile'] == 'Q1']
    q2_df = uncertainty_df[uncertainty_df['U_quartile'] == 'Q2']
    q3_df = uncertainty_df[uncertainty_df['U_quartile'] == 'Q3']
    q4_df = uncertainty_df[uncertainty_df['U_quartile'] == 'Q4']
    q5_df = uncertainty_df[uncertainty_df['U_quartile'] == 'Q5']
    
    q1_p_logit, _, _ = obtain_attn(q1_df, positive, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    
    q1_results = calculate_metric(q1_p_logit, q1_df.iloc[:, 6:-1].values, class_name, nf=True)
    print("Q1 Results:")
    print(q1_results)
    
    q2_p_logit, _, _ = obtain_attn(q2_df, positive, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    q2_results = calculate_metric(q2_p_logit, q2_df.iloc[:, 6:-1].values, class_name, nf=True)
    print("Q2 Results:")
    print(q2_results)
    
    q3_p_logit, _, _ = obtain_attn(q3_df, positive, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    q3_results = calculate_metric(q3_p_logit, q3_df.iloc[:, 6:-1].values, class_name, nf=True)
    print("Q3 Results:")
    print(q3_results)
    
    q4_p_logit, _, _ = obtain_attn(q4_df, positive, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    q4_results = calculate_metric(q4_p_logit, q4_df.iloc[:, 6:-1].values, class_name, nf=True)
    print("Q4 Results:")
    print(q4_results)
    
    q5_p_logit, _, _ = obtain_attn(q5_df, positive, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    q5_results = calculate_metric(q5_p_logit, q5_df.iloc[:, 6:-1].values, class_name, nf=True)
    print("Q5 Results:")
    print(q5_results)
    
    save_dir = os.path.join(args.result_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    q1_results.to_csv(os.path.join(save_dir, "NIH_Q1_pos_results.csv"))
    q2_results.to_csv(os.path.join(save_dir, "NIH_Q2_pos_results.csv"))
    q3_results.to_csv(os.path.join(save_dir, "NIH_Q3_pos_results.csv"))
    q4_results.to_csv(os.path.join(save_dir, "NIH_Q4_pos_results.csv"))
    q5_results.to_csv(os.path.join(save_dir, "NIH_Q5_pos_results.csv"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--tsne", type=str, default=False)
    args = parser.parse_args()
    
    main(args)