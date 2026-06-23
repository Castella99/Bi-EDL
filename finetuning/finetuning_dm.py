import os
from torch.utils.data import DataLoader
import torch
from glob import glob
from PIL import Image

from finetuning_dataset import *
import CARZero.builder as builder
import CARZero
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
import re

import random
from typing import List, Tuple

class NIHDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_df, val_df, test_df):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        print("Using NIHMCQOnlyDataset")
        if self.cfg.data.fewshot.enabled :
            fewshot_ratio = self.cfg.data.fewshot.ratio
            train_size = len(self.train_df)
            fewshot_size = int(train_size * fewshot_ratio)
            self.train_df = self.train_df.sample(n=fewshot_size, random_state=42).reset_index(drop=True)
            print(f"Few-shot enabled: Using {fewshot_size} samples out of {train_size} for training.")
        self.train_dataset = NIHDataset(self.train_df, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHDataset(self.val_df, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHDataset(self.test_df, self.cfg, transform=self.test_transform)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

class NIHMCQ2DataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_list, val_list, test_list):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        self.train_dataset = NIHMCQ2Dataset(self.root, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHMCQ2Dataset(self.root, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHMCQ2Dataset(self.root, self.cfg, transform=self.test_transform)

        self.train_dataset.df = self.train_dataset.df[self.train_dataset.df['Image Index'].isin(self.train_list)].reset_index(drop=True)
        self.val_dataset.df = self.val_dataset.df[self.val_dataset.df['Image Index'].isin(self.val_list)].reset_index(drop=True)
        self.test_dataset.df = self.test_dataset.df[self.test_dataset.df['Image Index'].isin(self.test_list)].reset_index(drop=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

class NIHMCQOnlyDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_df, val_df, test_df):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        print("Using NIHMCQOnlyDataset")
        if self.cfg.data.fewshot.enabled :
            fewshot_ratio = self.cfg.data.fewshot.ratio
            train_size = len(self.train_df)
            fewshot_size = int(train_size * fewshot_ratio)
            self.train_df = self.train_df.sample(n=fewshot_size, random_state=42).reset_index(drop=True)
            print(f"Few-shot enabled: Using {fewshot_size} samples out of {train_size} for training.")
        self.train_dataset = NIHMCQOnlyDataset(self.train_df, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHMCQOnlyDataset(self.val_df, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHMCQOnlyDataset(self.test_df, self.cfg, transform=self.test_transform)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )
        
class NIHMCQEDLDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_df, val_df, test_df):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        print("Using NIHMCQOnlyDataset")
        if self.cfg.data.fewshot.enabled :
            fewshot_ratio = self.cfg.data.fewshot.ratio
            train_size = len(self.train_df)
            fewshot_size = int(train_size * fewshot_ratio)
            self.train_df = self.train_df.sample(n=fewshot_size, random_state=42).reset_index(drop=True)
            print(f"Few-shot enabled: Using {fewshot_size} samples out of {train_size} for training.")
        self.train_dataset = NIHMCQEDLDataset(self.train_df, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHMCQEDLDataset(self.val_df, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHMCQEDLDataset(self.test_df, self.cfg, transform=self.test_transform)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )
        
class NIHMCQDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_list, val_list, test_list):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        self.train_dataset = NIHMCQDataset(self.root, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHMCQDataset(self.root, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHMCQDataset(self.root, self.cfg, transform=self.test_transform)
        
        self.train_dataset.df = self.train_dataset.df[self.train_dataset.df['Image Index'].isin(self.train_list)].reset_index(drop=True)
        self.val_dataset.df = self.val_dataset.df[self.val_dataset.df['Image Index'].isin(self.val_list)].reset_index(drop=True)
        self.test_dataset.df = self.test_dataset.df[self.test_dataset.df['Image Index'].isin(self.test_list)].reset_index(drop=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

class NIHMultiLabelMCQDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_list, val_list, test_list):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        self.train_dataset = NIHMultiLabelMCQDataset(self.root, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHMultiLabelMCQDataset(self.root, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHMultiLabelMCQDataset(self.root, self.cfg, transform=self.test_transform)
        
        self.train_dataset.df = self.train_dataset.df[self.train_dataset.df['Image Index'].isin(self.train_list)].reset_index(drop=True)
        self.val_dataset.df = self.val_dataset.df[self.val_dataset.df['Image Index'].isin(self.val_list)].reset_index(drop=True)
        self.test_dataset.df = self.test_dataset.df[self.test_dataset.df['Image Index'].isin(self.test_list)].reset_index(drop=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )
        
# class NIHDataModule(pl.LightningDataModule):
#     def __init__(self, cfg, root, train_list, val_list, test_list):
#         super().__init__()
#         self.cfg = cfg
#         self.root = root
#         self.train_list = train_list
#         self.val_list = val_list
#         self.test_list = test_list
#         self.train_transform = builder.build_transformation(cfg, 'train')
#         self.test_transform = builder.build_transformation(cfg, 'test')

#     def setup(self, stage=None):
#         if self.cfg.data.low_uncertainty :
#             self.train_dataset = NIHPromptDataset(self.root, self.cfg, transform=self.train_transform, df=self.cfg.data.train.df_name)
#             self.val_dataset = NIHPromptDataset(self.root, self.cfg, transform=self.test_transform, df=self.cfg.data.val.df_name)
#             self.test_dataset = NIHPromptDataset(self.root, self.cfg, transform=self.test_transform, df=self.cfg.data.test.df_name)

#         else :
#             self.train_dataset = NIHPromptDataset(self.root, self.cfg, transform=self.train_transform)
#             self.val_dataset = NIHPromptDataset(self.root, self.cfg, transform=self.test_transform)
#             self.test_dataset = NIHPromptDataset(self.root, self.cfg, transform=self.test_transform)
            
#             self.train_dataset.df = self.train_dataset.df[self.train_dataset.df['Image Index'].isin(self.train_list)].reset_index(drop=True)
#             self.val_dataset.df = self.val_dataset.df[self.val_dataset.df['Image Index'].isin(self.val_list)].reset_index(drop=True)
#             self.test_dataset.df = self.test_dataset.df[self.test_dataset.df['Image Index'].isin(self.test_list)].reset_index(drop=True)
        
#         print(f"Train dataset size: {len(self.train_dataset)}")
#         print(f"Val dataset size: {len(self.val_dataset)}")
#         print(f"Test dataset size: {len(self.test_dataset)}")

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.cfg.train.batch_size,
#             shuffle=True,
#             num_workers=self.cfg.train.num_workers,
#             pin_memory=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.cfg.test.batch_size,
#             shuffle=False,
#             num_workers=self.cfg.test.num_workers,
#             pin_memory=True
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.cfg.test.batch_size,
#             shuffle=False,
#             num_workers=self.cfg.test.num_workers,
#             pin_memory=True
#         )

class NIHMultiLabelMCQDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_list, val_list, test_list):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        self.train_dataset = NIHMultiLabelMCQDataset(self.root, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHMultiLabelMCQDataset(self.root, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHMultiLabelMCQDataset(self.root, self.cfg, transform=self.test_transform)
        
        self.train_dataset.df = self.train_dataset.df[self.train_dataset.df['Image Index'].isin(self.train_list)].reset_index(drop=True)
        self.val_dataset.df = self.val_dataset.df[self.val_dataset.df['Image Index'].isin(self.val_list)].reset_index(drop=True)
        self.test_dataset.df = self.test_dataset.df[self.test_dataset.df['Image Index'].isin(self.test_list)].reset_index(drop=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

class NIHPromptDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_list, val_list, test_list, neg=True):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')
        self.neg = neg

    def setup(self, stage=None):
        self.train_dataset = NIHDataset(self.root, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHDataset(self.root, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHDataset(self.root, self.cfg, transform=self.test_transform)
        
        self.train_dataset.df = self.train_dataset.df[self.train_dataset.df['Image Index'].isin(self.train_list)].reset_index(drop=True)
        self.val_dataset.df = self.val_dataset.df[self.val_dataset.df['Image Index'].isin(self.val_list)].reset_index(drop=True)
        self.test_dataset.df = self.test_dataset.df[self.test_dataset.df['Image Index'].isin(self.test_list)].reset_index(drop=True)

        # ---- create collate functions (single instances) ----
        self.train_collate = PromptBatchCollator(
            class_names=self.train_dataset.class_names,
            tokenizer=self.train_dataset.tokenizer,
            seq_len=self.cfg.data.text.word_num,
            prob_sched=linear_prob_scheduler(self.cfg.train.max_epoch) if self.cfg.train.prob_sched else None
        )
        self.val_collate = PromptBatchCollator(
            class_names=self.val_dataset.class_names,
            tokenizer=self.val_dataset.tokenizer,
            seq_len=self.cfg.data.text.word_num
        )
        self.test_collate = PromptBatchCollator(
            class_names=self.test_dataset.class_names,
            tokenizer=self.test_dataset.tokenizer,
            seq_len=self.cfg.data.text.word_num
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            collate_fn=self.train_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True,
            collate_fn=self.val_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True,
            collate_fn=self.test_collate
        )

class NIHPNPromptDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_list, val_list, test_list):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        self.train_dataset = NIHDataset(self.root, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHDataset(self.root, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHDataset(self.root, self.cfg, transform=self.test_transform)
        
        self.train_dataset.df = self.train_dataset.df[self.train_dataset.df['Image Index'].isin(self.train_list)].reset_index(drop=True)
        self.val_dataset.df = self.val_dataset.df[self.val_dataset.df['Image Index'].isin(self.val_list)].reset_index(drop=True)
        self.test_dataset.df = self.test_dataset.df[self.test_dataset.df['Image Index'].isin(self.test_list)].reset_index(drop=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=lambda batch: nih_collate_with_truth(
                batch,
                self.train_dataset.tokenizer,
                self.train_dataset.class_names
            ),
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: nih_collate_with_truth(
                batch,
                self.train_dataset.tokenizer,
                self.train_dataset.class_names
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: nih_collate_with_truth(
                batch,
                self.train_dataset.tokenizer,
                self.train_dataset.class_names
            ),
        )
        
class NIHDCLPromptDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_list, val_list, test_list):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        self.train_dataset = NIHDCLDataset(self.root, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHDCLDataset(self.root, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHDCLDataset(self.root, self.cfg, transform=self.test_transform)
        
        self.train_dataset.df = self.train_dataset.df[self.train_dataset.df['Image Index'].isin(self.train_list)].reset_index(drop=True)
        self.val_dataset.df = self.val_dataset.df[self.val_dataset.df['Image Index'].isin(self.val_list)].reset_index(drop=True)
        self.test_dataset.df = self.test_dataset.df[self.test_dataset.df['Image Index'].isin(self.test_list)].reset_index(drop=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate_fn,
        )
        
class NIHPosNegDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_df, val_df, test_df):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        print("Using NIHPosNegDataset")
        # if self.cfg.data.fewshot.enabled :
        #     fewshot_ratio = self.cfg.data.fewshot.ratio
        #     train_size = len(self.train_df)
        #     fewshot_size = int(train_size * fewshot_ratio)
        #     self.train_df = self.train_df.sample(n=fewshot_size, random_state=42).reset_index(drop=True)
        #     print(f"Few-shot enabled: Using {fewshot_size} samples out of {train_size} for training.")
        self.train_dataset = NIHPosNegDataset(self.train_df, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHPosNegDataset(self.val_df, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHPosNegDataset(self.test_df, self.cfg, transform=self.test_transform)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )