import sys
import os
os.chdir('/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero')
sys.path.append('/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero')

import torch
import CARZero
import pandas as pd
import numpy as np
from utils import *
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from uncertainty_utils import *
from finetuning_inference import obtain_simr, calculate_pnc_logit, calculate_metric
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.3f}'.format
plt.style.use('default')

class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']

positive = {"0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Pleural Effusion"], "3": ["There is Pulmonary Infiltration"], "4": ["There is Pulmonary Mass"], "5": ["There is Lung Nodule"], "6": ["There is Pneumonia"], "7": ["There is Pneumothorax"], "8": ["There is Pulmonary Consolidation"], "9": ["There is Pulmonary Edema"], "10": ["There is Pulmonary Emphysema"], "11": ["There is Fibrosis"], "12": ["There is Pleural Thickening"], "13": ["There is Hernia"], "14" : ["There is no Finding"]}
negative = {"0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Pleural Effusion"], "3": ["There is no Pulmonary Infiltration"], "4": ["There is no Pulmonary Mass"], "5": ["There is no Lung Nodule"], "6": ["There is no Pneumonia"], "7": ["There is no Pneumothorax"], "8": ["There is no Pulmonary Consolidation"], "9": ["There is no Pulmonary Edema"], "10": ["There is no Pulmonary Emphysema"], "11": ["There is no Fibrosis"], "12": ["There is no Pleural Thickening"], "13": ["There is no Hernia"]}

prompts = ["There is Atelectasis", "There is Cardiomegaly", "There is Pleural Effusion", "There is Pulmonary Infiltration", "There is Pulmonary Mass", "There is Lung Nodule", "There is Pneumonia", "There is Pneumothorax", "There is Pulmonary Consolidation", "There is Pulmonary Edema", "There is Pulmonary Emphysema", "There is Fibrosis", "There is Pleural Thickening", "There is Hernia", "There is no Finding", "There is no Atelectasis", "There is no Cardiomegaly", "There is no Pleural Effusion", "There is no Pulmonary Infiltration", "There is no Pulmonary Mass", "There is no Lung Nodule", "There is no Pneumonia", "There is no Pneumothorax", "There is no Pulmonary Consolidation", "There is no Pulmonary Edema", "There is no Pulmonary Emphysema", "There is no Fibrosis", "There is no Pleural Thickening", "There is no Hernia"]

def main(args):
    cfg = OmegaConf.load(os.path.join(args.result_dir, 'config.yaml')) if args.result_dir else None
    if not args.result_dir:
        args.result_dir = "./logs/CARZero_Zeroshot/"
        cfg = OmegaConf.load(os.path.join(args.result_dir, 'config.yaml'))
        CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device="cuda", multi=False, cfg=cfg)
    else:
        checkpoint = os.path.join(args.result_dir, "checkpoints/best_model.ckpt")
        CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device="cuda", multi=cfg.model.CARZero.multi, cfg=cfg)
        ckpt_state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
        fixed_ckpt_dict = {k.split("CARZero_model.")[-1]: v for k, v in ckpt_state_dict.items() if k.split("CARZero_model.")[-1] in CARZero_model.state_dict()}
        CARZero_model.load_state_dict(fixed_ckpt_dict, strict=True)
    CARZero_model.eval()

    print(args.result_dir)

    data_path = '/shared/home/mai/Taehun/Uncertainty/data/NIH'
    label_path = 'ChestXray-14'
    csv_head = ['path', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Lung Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']
    label_file_path = os.path.join(label_path, 'test_list.txt')
    df_test = pd.read_csv(label_file_path, sep=' ', names=csv_head)
    key = csv_head[1:]
    df_test['No Finding'] = (df_test[key].sum(axis=1) == 0).astype(int)
    key = key + ['No Finding']
    df_test['Image Index'] = df_test['path'].apply(lambda x: os.path.basename(x))
    if 'Image Index' in df_test.columns:
        df_test.insert(0, 'Image Index', df_test.pop('Image Index'))
    img_path = {os.path.basename(x): x for x in glob(os.path.join(data_path, 'images*', '*', '*.png'))}
    df_test['path'] = df_test['Image Index'].map(img_path)
    rename_map = {'path': 'Path', 'Lung Mass': 'Mass', 'Lung Nodule': 'Nodule'}
    test_df = df_test.rename(columns=rename_map)
    # test_df = test_df.iloc[:100]

    true_labels = test_df.iloc[:, 2:].values

    save_dir = os.path.join(args.result_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    # Single inference pass: logits + i2t/t2i features simultaneously
    logit, i2t_feat, t2i_feat = obtain_simr(test_df, prompts, CARZero_model, mode="both", device="cuda:0", multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi, ts=True)
    p_logit = logit[:, :len(class_name)]
    n_logit = logit[:, len(class_name):]
    p_i2t_feat = i2t_feat[:, :len(class_name),:]
    n_i2t_feat = i2t_feat[:, len(class_name):,:]
    p_t2i_feat = t2i_feat[:, :len(class_name),:]
    n_t2i_feat = t2i_feat[:, len(class_name):,:]
    # p_logit, p_i2t_feat, p_t2i_feat = obtain_simr(test_df, positive, CARZero_model, device="cuda:0", mode='i2t', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
    # n_logit, n_i2t_feat, n_t2i_feat = obtain_simr(test_df, negative, CARZero_model, device="cuda:0", mode='i2t', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)

    np.save(os.path.join(args.result_dir, "logit.npy"), logit)
    np.save(os.path.join(args.result_dir, "p_i2t_feat.npy"), p_i2t_feat)
    np.save(os.path.join(args.result_dir, "n_i2t_feat.npy"), n_i2t_feat)
    np.save(os.path.join(args.result_dir, "p_t2i_feat.npy"), p_t2i_feat)
    np.save(os.path.join(args.result_dir, "n_t2i_feat.npy"), n_t2i_feat)

    pnc_logit = calculate_pnc_logit(p_logit[:, :-1], n_logit)

    pos_results = calculate_metric(p_logit, true_labels, class_name, nf=True)
    neg_results = calculate_metric(n_logit, true_labels, class_name, neg=True)
    pnc_results = calculate_metric(pnc_logit, true_labels, class_name)

    print("Positive Results:")
    print(pos_results)
    print("Negative Results:")
    print(neg_results)
    print("Positive-Negative Combined Results:")
    print(pnc_results)

    pos_results.to_csv(os.path.join(save_dir, "NIH_pos_results.csv"))
    neg_results.to_csv(os.path.join(save_dir, "NIH_neg_results.csv"))
    pnc_results.to_csv(os.path.join(save_dir, "NIH_pnc_results.csv"))

    if args.dir:
        p_logit_t2i, _, _ = obtain_simr(test_df, positive, CARZero_model, device="cuda:0", mode='t2i', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)
        n_logit_t2i, _, _ = obtain_simr(test_df, negative, CARZero_model, device="cuda:0", mode='t2i', multi=cfg.model.CARZero.multi, mcq=cfg.model.CARZero.multi)

        pnc_i2t_logit = calculate_pnc_logit(p_logit[:, :-1], n_logit)
        pnc_t2i_logit = calculate_pnc_logit(p_logit_t2i[:, :-1], n_logit_t2i)

        pos_i2t_results = calculate_metric(p_logit, true_labels, class_name, nf=True)
        neg_i2t_results = calculate_metric(n_logit, true_labels, class_name, neg=True)
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

    if args.tsne:
        n_classes = n_logit.shape[1]

        pn_i2t_features_tsne = np.array([
            TSNE(n_components=2, random_state=42).fit_transform(
                np.concatenate((p_i2t_feat[:, i, :], n_i2t_feat[:, i, :]), axis=0)
            ) for i in tqdm(range(n_classes), desc="TSNE i2t")
        ])
        pn_t2i_features_tsne = np.array([
            TSNE(n_components=2, random_state=42).fit_transform(
                np.concatenate((p_t2i_feat[:, i, :], n_t2i_feat[:, i, :]), axis=0)
            ) for i in tqdm(range(n_classes), desc="TSNE t2i")
        ])

        p_i2t_feat_tsne = pn_i2t_features_tsne[:, :p_i2t_feat.shape[0], :]
        n_i2t_feat_tsne = pn_i2t_features_tsne[:, p_i2t_feat.shape[0]:, :]
        p_t2i_feat_tsne = pn_t2i_features_tsne[:, :p_t2i_feat.shape[0], :]
        n_t2i_feat_tsne = pn_t2i_features_tsne[:, p_t2i_feat.shape[0]:, :]

        plt.figure(figsize=(20, 20))
        for i in range(n_classes):
            true_sample = (test_df.iloc[:, i + 3] == 1)
            false_sample = (test_df.iloc[:, i + 3] == 0)
            plt.subplot(4, 4, i + 1)
            plt.scatter(p_i2t_feat_tsne[i][false_sample, 0], p_i2t_feat_tsne[i][false_sample, 1], s=0.5, c='red', label='Positive - False', alpha=0.5)
            plt.scatter(n_i2t_feat_tsne[i][false_sample, 0], n_i2t_feat_tsne[i][false_sample, 1], s=0.5, c='green', label='Negative - False', alpha=0.5)
            plt.scatter(p_i2t_feat_tsne[i][true_sample, 0], p_i2t_feat_tsne[i][true_sample, 1], s=1, c='blue', label='Positive - True', alpha=0.5)
            plt.scatter(n_i2t_feat_tsne[i][true_sample, 0], n_i2t_feat_tsne[i][true_sample, 1], s=1, c='orange', label='Negative - True', alpha=0.5)
            plt.xlabel('t-SNE Component 1')
            plt.title(f"Class - {class_name[i]}, PosAUC - {pos_results.loc[class_name[i], 'auc']:.3f}, NegAUC - {neg_results.loc[class_name[i], 'auc']:.3f}")
            plt.legend()
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/NIH_i2t_tsne_plot.png")
        plt.close()

        plt.figure(figsize=(20, 20))
        for i in range(n_classes):
            true_sample = (test_df.iloc[:, i + 3] == 1)
            false_sample = (test_df.iloc[:, i + 3] == 0)
            plt.subplot(4, 4, i + 1)
            plt.scatter(p_t2i_feat_tsne[i][false_sample, 0], p_t2i_feat_tsne[i][false_sample, 1], s=0.5, c='red', label='Positive - False', alpha=0.5)
            plt.scatter(n_t2i_feat_tsne[i][false_sample, 0], n_t2i_feat_tsne[i][false_sample, 1], s=0.5, c='green', label='Negative - False', alpha=0.5)
            plt.scatter(p_t2i_feat_tsne[i][true_sample, 0], p_t2i_feat_tsne[i][true_sample, 1], s=1, c='blue', label='Positive - True', alpha=0.5)
            plt.scatter(n_t2i_feat_tsne[i][true_sample, 0], n_t2i_feat_tsne[i][true_sample, 1], s=1, c='orange', label='Negative - True', alpha=0.5)
            plt.xlabel('t-SNE Component 1')
            plt.title(f"Class - {class_name[i]}, PosAUC - {pos_results.loc[class_name[i], 'auc']:.3f}, NegAUC - {neg_results.loc[class_name[i], 'auc']:.3f}")
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
