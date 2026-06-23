import os
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

from finetuning_lightening import DCLModel
from finetuning_dm import NIHDCLPromptDataModule

def main(cfg) :
    data_path = '/shared/home/mai/Taehun/Uncertainty/data/NIH'
    with open(os.path.join(data_path, 'train_val_list.txt'), 'r') as f :
        train_val_list = f.readlines()
    with open(os.path.join(data_path, 'test_list.txt'), 'r') as f :
        test_list = f.readlines()
    train_val_list = [x.strip() for x in train_val_list]
    test_list = [x.strip() for x in test_list]

    train_list, val_list = train_test_split(train_val_list, test_size=0.2, random_state=cfg.train.seed)

    dm = NIHDCLPromptDataModule(
        cfg,
        root=data_path,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
    )

    model = DCLModel(cfg)

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
                monitor=cfg.lightning.checkpoint_callback.monitor,
                dirpath=os.path.join(log_path, "checkpoints"),
                filename="best_model",
                save_top_k=1,
                mode=cfg.lightning.checkpoint_callback.mode,
            ),
            EarlyStopping(monitor=cfg.lightning.early_stopping_callback.monitor, patience=cfg.lightning.early_stopping_callback.patience, mode=cfg.lightning.early_stopping_callback.mode),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=None,
    )

    trainer.save_checkpoint(os.path.join(log_path, "final_model.ckpt"))
    
    # 실험 종료 후 테스트 수행
    print("Evaluating on test set...")
    trainer.test(model, datamodule=dm)

if __name__ == "__main__" :
    cfg = OmegaConf.load('configs/chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_DCL.yaml')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)
    main(cfg)
