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
from finetuning_inference import obtain_attn, obtain_simr, calculate_pnc_logit, calculate_metric
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:.3f}'.format

plt.style.use('default')

class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']

positive = {"0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Pleural Effusion"], "3": ["There is Pulmonary Infiltration"], "4": ["There is Pulmonary Mass"], "5": ["There is Lung Nodule"], "6": ["There is Pneumonia"], "7": ["There is Pneumothorax"], "8": ["There is Pulmonary Consolidation"], "9": ["There is Pulmonary Edema"], "10": ["There is Pulmonary Emphysema"], "11": ["There is Fibrosis"], "12": ["There is Pleural Thickening"], "13": ["There is Hernia"], "14" : ["There is no Finding"]}
negative = {"0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Pleural Effusion"], "3": ["There is no Pulmonary Infiltration"], "4": ["There is no Pulmonary Mass"], "5": ["There is no Lung Nodule"], "6": ["There is no Pneumonia"], "7": ["There is no Pneumothorax"], "8": ["There is no Pulmonary Consolidation"], "9": ["There is no Pulmonary Edema"], "10": ["There is no Pulmonary Emphysema"], "11": ["There is no Fibrosis"], "12": ["There is no Pleural Thickening"], "13": ["There is no Hernia"]}

prompts = ["There is Atelectasis", "There is Cardiomegaly", "There is Pleural Effusion", "There is Pulmonary Infiltration", "There is Pulmonary Mass", "There is Lung Nodule", "There is Pneumonia", "There is Pneumothorax", "There is Pulmonary Consolidation", "There is Pulmonary Edema", "There is Pulmonary Emphysema", "There is Fibrosis", "There is Pleural Thickening", "There is Hernia", "There is no Finding", "There is no Atelectasis", "There is no Cardiomegaly", "There is no Pleural Effusion", "There is no Pulmonary Infiltration", "There is no Pulmonary Mass", "There is no Lung Nodule", "There is no Pneumonia", "There is no Pneumothorax", "There is no Pulmonary Consolidation", "There is no Pulmonary Edema", "There is no Pulmonary Emphysema", "There is no Fibrosis", "There is no Pleural Thickening", "There is no Hernia"]

pathologies = [
        # NIH
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        "No_Finding",
    ]

mapping = dict()
mapping["Pleural_Thickening"] = ["pleural thickening"]
mapping["Infiltration"] = ["Infiltrate"]
mapping["Atelectasis"] = ["Atelectases"]
mapping["No_Finding"] = ["-1"]

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
    
    data_path = "/shared/home/mai/Taehun/Uncertainty/data/OPEN-I"
    csv = pd.read_csv(os.path.join(data_path, 'custom.csv')).iloc[2:, :].reset_index(drop=True)
    csv = csv.replace(np.nan, "-1")
    
    gt = []
    for pathology in pathologies:
        mask = csv["labels_automatic"].str.contains(pathology.lower())
        if pathology in mapping:
            for syn in mapping[pathology]:
                # print("mapping", syn)
                mask |= csv["labels_automatic"].str.contains(syn.lower())
        gt.append(mask.values)
        
    gt = np.asarray(gt).T
    gt = gt.astype(np.float32)
    gt[:, 14] = (gt[:, :14].sum(axis=1) == 0).astype(np.float32)
    
    image_csv = pd.read_csv(os.path.join(data_path, 'openi_multi_label_image.csv'), ).iloc[2:, :].reset_index(drop=True)
    image_csv['Path'] = image_csv['Path'].apply(lambda x: os.path.join(data_path, x.split('/')[-1]))

    logit, i2t_feat, t2i_feat = obtain_simr(image_csv, prompts, CARZero_model, mode="both", device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi, ts=True)
    p_logit = logit[:, :len(class_name)]
    n_logit = logit[:, len(class_name):]
    p_i2t_feat = i2t_feat[:, :len(class_name),:]
    n_i2t_feat = i2t_feat[:, len(class_name):,:]
    p_t2i_feat = t2i_feat[:, :len(class_name),:]
    n_t2i_feat = t2i_feat[:, len(class_name):,:]
    
    pnc_logit = calculate_pnc_logit(p_logit[:,:-1], n_logit)
    
    pos_results = calculate_metric(p_logit, gt, class_name, nf=True)
    neg_results = calculate_metric(n_logit, gt, class_name, neg=True)
    pnc_results = calculate_metric(pnc_logit, gt, class_name)
    
    print("Positive Results:")
    print(pos_results)
    print("Negative Results:")
    print(neg_results)
    print("Positive-Negative Combined Results:")
    print(pnc_results)
    
    save_dir = os.path.join(args.result_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    pos_results.to_csv(os.path.join(save_dir, "OPEN_I_pos_results.csv"))
    neg_results.to_csv(os.path.join(save_dir, "OPEN_I_neg_results.csv"))
    pnc_results.to_csv(os.path.join(save_dir, "OPEN_I_pnc_results.csv"))
    
    if args.dir :
        p_logit_i2t, p_i2t_feat, p_t2i_feat = obtain_simr(image_csv, positive, CARZero_model, device="cuda:0", mode='i2t', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        n_logit_i2t, n_i2t_feat, n_t2i_feat = obtain_simr(image_csv, negative, CARZero_model, device="cuda:0", mode='i2t', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        
        p_logit_t2i, p_i2t_feat, p_t2i_feat = obtain_simr(image_csv, positive, CARZero_model, device="cuda:0", mode='t2i', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        n_logit_t2i, n_i2t_feat, n_t2i_feat = obtain_simr(image_csv, negative, CARZero_model, device="cuda:0", mode='t2i', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        
        pnc_i2t_logit = calculate_pnc_logit(p_logit_i2t[:, :-1], n_logit_i2t)
        pnc_t2i_logit = calculate_pnc_logit(p_logit_t2i[:, :-1], n_logit_t2i)
        
        pos_i2t_results = calculate_metric(p_logit_i2t, gt, class_name, nf=True)
        neg_i2t_results = calculate_metric(n_logit_i2t, gt, class_name, neg=True)
        pnc_i2t_results = calculate_metric(pnc_i2t_logit, gt, class_name)
        
        pos_t2i_results = calculate_metric(p_logit_t2i, gt, class_name, nf=True)
        neg_t2i_results = calculate_metric(n_logit_t2i, gt, class_name, neg=True)
        pnc_t2i_results = calculate_metric(pnc_t2i_logit, gt, class_name)
        
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
        
        pos_i2t_results.to_csv(os.path.join(save_dir, "OPEN_I_pos_i2t_results.csv"))
        neg_i2t_results.to_csv(os.path.join(save_dir, "OPEN_I_neg_i2t_results.csv"))
        pnc_i2t_results.to_csv(os.path.join(save_dir, "OPEN_I_pnc_i2t_results.csv"))
        
        pos_t2i_results.to_csv(os.path.join(save_dir, "OPEN_I_pos_t2i_results.csv"))
        neg_t2i_results.to_csv(os.path.join(save_dir, "OPEN_I_neg_t2i_results.csv"))
        pnc_t2i_results.to_csv(os.path.join(save_dir, "OPEN_I_pnc_t2i_results.csv")) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--tsne", type=str, default=False)
    parser.add_argument("--dir", type=str, default=False)
    args = parser.parse_args()
    
    main(args)