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

pd.options.display.float_format = '{:.3f}'.format

plt.style.use('default')

class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']

positive = {"0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Pleural Effusion"], "3": ["There is Pulmonary Infiltration"], "4": ["There is Pulmonary Mass"], "5": ["There is Lung Nodule"], "6": ["There is Pneumonia"], "7": ["There is Pneumothorax"], "8": ["There is Pulmonary Consolidation"], "9": ["There is Pulmonary Edema"], "10": ["There is Pulmonary Emphysema"], "11": ["There is Fibrosis"], "12": ["There is Pleural Thickening"], "13": ["There is Hernia"], "14" : ["There is no Finding"]}
negative = {"0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Pleural Effusion"], "3": ["There is no Pulmonary Infiltration"], "4": ["There is no Pulmonary Mass"], "5": ["There is no Lung Nodule"], "6": ["There is no Pneumonia"], "7": ["There is no Pneumothorax"], "8": ["There is no Pulmonary Consolidation"], "9": ["There is no Pulmonary Edema"], "10": ["There is no Pulmonary Emphysema"], "11": ["There is no Fibrosis"], "12": ["There is no Pleural Thickening"], "13": ["There is no Hernia"]}

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
    
    p_logit, p_i2t_feat, p_t2i_feat = obtain_simr(test_df, positive, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi, mode='both')
    n_logit, n_i2t_feat, n_t2i_feat = obtain_simr(test_df, negative, CARZero_model, device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi, mode='both')
    
    save_dir = os.path.join(args.result_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    pn_i2t_features_tsne = [TSNE(n_components=2, random_state=42).fit_transform(np.concatenate((p_i2t_feat[:,i,:], n_i2t_feat[:,i,:]), axis=0)) for i in tqdm(range(n_logit.shape[1]))]
    pn_t2i_features_tsne = [TSNE(n_components=2, random_state=42).fit_transform(np.concatenate((p_t2i_feat[:,i,:], n_t2i_feat[:,i,:]), axis=0)) for i in tqdm(range(n_logit.shape[1]))]

    pn_i2t_features_tsne = np.array(pn_i2t_features_tsne)
    pn_t2i_features_tsne = np.array(pn_t2i_features_tsne)

    p_i2t_feat_tsne = pn_i2t_features_tsne[:, :p_i2t_feat.shape[0], :]
    p_t2i_feat_tsne = pn_t2i_features_tsne[:, :p_t2i_feat.shape[0], :]
    n_i2t_feat_tsne = pn_i2t_features_tsne[:, p_i2t_feat.shape[0]:, :]
    n_t2i_feat_tsne = pn_t2i_features_tsne[:, p_t2i_feat.shape[0]:, :]
    
    np.save(f"{save_dir}/pn_i2t_features_tsne.npy", pn_i2t_features_tsne)
    np.save(f"{save_dir}/pn_t2i_features_tsne.npy", pn_t2i_features_tsne)
    
    plt.figure(figsize=(20, 20))
    for i in range(n_logit.shape[1]):
        true_sample = (test_df.iloc[:,i+3]==1)
        false_sample = (test_df.iloc[:,i+3]==0)

        plt.subplot(4, 4, i+1)
        plt.scatter(p_i2t_feat_tsne[i][false_sample, 0], p_i2t_feat_tsne[i][false_sample, 1], s=0.5, c='red', label='Positive - False', alpha=0.5)
        plt.scatter(n_i2t_feat_tsne[i][false_sample, 0], n_i2t_feat_tsne[i][false_sample, 1], s=0.5, c='green', label='Negative - False', alpha=0.5)
        plt.scatter(p_i2t_feat_tsne[i][true_sample, 0], p_i2t_feat_tsne[i][true_sample, 1], s=1, c='blue', label='Positive - True', alpha=0.5)
        plt.scatter(n_i2t_feat_tsne[i][true_sample, 0], n_i2t_feat_tsne[i][true_sample, 1], s=1, c='orange', label='Negative - True', alpha=0.5)
        plt.xlabel('TSNE Component 1')
        plt.title(f"Class - {class_name[i]}")
        plt.legend()
        plt.axis('off')   
    plt.tight_layout()
    plt.savefig(f"{save_dir}/NIH_i2t_tsne_plot.png")
    plt.show()

    plt.figure(figsize=(20, 20))
    for i in range(n_logit.shape[1]):
        true_sample = (test_df.iloc[:,i+3]==1)
        false_sample = (test_df.iloc[:,i+3]==0)
        plt.subplot(4, 4, i+1)
        plt.scatter(p_t2i_feat_tsne[i][false_sample, 0], p_t2i_feat_tsne[i][false_sample, 1], s=0.5, c='red', label='Positive - False', alpha=0.5)
        plt.scatter(n_t2i_feat_tsne[i][false_sample, 0], n_t2i_feat_tsne[i][false_sample, 1], s=0.5, c='green', label='Negative - False', alpha=0.5)
        plt.scatter(p_t2i_feat_tsne[i][true_sample, 0], p_t2i_feat_tsne[i][true_sample, 1], s=1, c='blue', label='Positive - True', alpha=0.5)
        plt.scatter(n_t2i_feat_tsne[i][true_sample, 0], n_t2i_feat_tsne[i][true_sample, 1], s=1, c='orange', label='Negative - True', alpha=0.5)
        plt.xlabel('TSNE Component 1')
        plt.title(f"Class - {class_name[i]}")
        plt.legend()
        plt.axis('off')   
    plt.tight_layout()
    plt.savefig(f"{save_dir}/NIH_t2i_tsne_plot.png")
    plt.show()
        
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=None)
    args = parser.parse_args()
    
    main(args)