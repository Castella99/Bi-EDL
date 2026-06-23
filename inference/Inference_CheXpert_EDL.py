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

class_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'No Finding']

positive = {"0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Consolidation"], "3": ["There is Edema"], "4": ["There is Pleural Effusion"], "5" : ["There is no Finding"]}
negative = {"0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Consolidation"], "3": ["There is no Edema"], "4": ["There is no Pleural Effusion"]}

prompts = ["There is Atelectasis", "There is Cardiomegaly", "There is Consolidation", "There is Edema", "There is Pleural Effusion", "There is no Finding", "There is no Atelectasis", "There is no Cardiomegaly", "There is no Consolidation", "There is no Edema", "There is no Pleural Effusion"]

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
    
    data_path = "/shared/home/mai/Taehun/Uncertainty/data/Chestpert/chexlocalize/CheXpert"
    image_csv = pd.read_csv(os.path.join("Chexpert", 'chexpert5_test_image.csv'), )
    image_csv['Path'] = image_csv['Path'].apply(lambda x: os.path.join(data_path, '/'.join(x.split('/')[3:])))
    label_csv = pd.read_csv(os.path.join("Chexpert", 'test_labels.csv'), )
    label_csv = label_csv[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'No Finding']]
    gt = label_csv.values
    
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
    
    pos_results.to_csv(os.path.join(save_dir, "CheXpert_pos_results.csv"))
    neg_results.to_csv(os.path.join(save_dir, "CheXpert_neg_results.csv"))
    pnc_results.to_csv(os.path.join(save_dir, "CheXpert_pnc_results.csv"))
    
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
        
        pos_i2t_results.to_csv(os.path.join(save_dir, "CheXpert_pos_i2t_results.csv"))
        neg_i2t_results.to_csv(os.path.join(save_dir, "CheXpert_neg_i2t_results.csv"))
        pnc_i2t_results.to_csv(os.path.join(save_dir, "CheXpert_pnc_i2t_results.csv"))
        
        pos_t2i_results.to_csv(os.path.join(save_dir, "CheXpert_pos_t2i_results.csv"))
        neg_t2i_results.to_csv(os.path.join(save_dir, "CheXpert_neg_t2i_results.csv"))
        pnc_t2i_results.to_csv(os.path.join(save_dir, "CheXpert_pnc_t2i_results.csv")) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--tsne", type=str, default=False)
    parser.add_argument("--dir", type=str, default=False)
    args = parser.parse_args()
    
    main(args)