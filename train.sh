#!/bin/bash

DATA_PATH="../data/NIH"
CFG_PATH="configs/chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_Bi_EDL.yaml"

python train.py \
    --data_path "$DATA_PATH" \
    --cfg_path "$CFG_PATH"
