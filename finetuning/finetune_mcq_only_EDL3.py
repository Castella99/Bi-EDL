import os
import sys
sys.path.append("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero")
sys.path.append("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero/finetuning")
os.chdir("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero")
import torch
import CARZero
import pandas as pd 
import json
import numpy as np
from utils import *
from sklearn.preprocessing import MultiLabelBinarizer
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from dateutil import tz
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import CARZero.builder as builder
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from finetuning_lightening import MCQEDLLightModel
from finetuning_dm import NIHDataModule

def main(cfg) :
    class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']
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
    label_file_path = os.path.join('ChestXray-14', 'test_list.txt')
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
    df_test = df_test.rename(columns=rename_map)

    df_train = df[~df['Image Index'].isin(df_test['Image Index'])].reset_index(drop=True)
    df_train = df_train.drop(columns=['Finding Labels'], errors='ignore')
    
    train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=42)

    dm = NIHDataModule(
        cfg,
        root=data_path,
        train_df=train_df,
        val_df=df_test,
        test_df=df_test,
    )

    model = MCQEDLLightModel(cfg)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", cfg.project, cfg.name, timestamp)
    os.makedirs(log_path, exist_ok=True)
    config_save_path = os.path.join(log_path, "config.yaml")
    OmegaConf.save(cfg, config_save_path)

    wandb_logger = WandbLogger(
        project=cfg.project,      # 프로젝트 이름
        name=cfg.name,            # 실험 이름
        save_dir=log_path,        # 저장 경로
        log_model=True                              # 모델 저장 여부
    )

    trainer = Trainer(
        precision=cfg.lightning.trainer.precision,
        accelerator="gpu",
        devices=cfg.lightning.trainer.gpus,
        max_epochs=cfg.lightning.trainer.max_epochs,
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(log_path, "checkpoints"),
                filename="model-{epoch:02d}",
                every_n_epochs=1,
                save_top_k=-1,  
            ),
            ModelCheckpoint(
            monitor="val/mean_auroc",
            dirpath=os.path.join(log_path, "checkpoints", "best"),
            filename="best_model",
            save_top_k=1,
            mode="max",),
            EarlyStopping(monitor="val/mean_auroc", patience=10, mode="max"),
            LearningRateMonitor(logging_interval="step"),
        ],
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=None,
    )

    # 실험 종료 후 테스트 수행
    print("Evaluating on test set...")
    trainer.test(model, datamodule=dm)

if __name__ == "__main__" :
    cfg = OmegaConf.load('configs/chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_MCQ_Only_EDL.yaml')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)
    main(cfg)
