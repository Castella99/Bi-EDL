#!/bin/bash

CKPT_PATH="checkpoints/best_model.ckpt"
CFG_PATH="configs/chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_Bi_EDL.yaml"
DATA_PATH="../data/NIH"
DEVICE="cuda:0"
BATCH_SIZE=128
COVERAGE=0.9
ODIN_EPS=0.001

# 실행할 불확실성 방법 선택 (msp / energy / odin / maxlogit / edl)
METHOD="msp energy maxlogit edl"
# METHOD="edl"
# METHOD="msp energy edl"

python inference.py \
    --ckpt_path  "$CKPT_PATH" \
    --cfg_path   "$CFG_PATH" \
    --data_path  "$DATA_PATH" \
    --method     $METHOD \
    --device     "$DEVICE" \
    --batch_size $BATCH_SIZE \
    --coverage   $COVERAGE \
    --odin_eps   $ODIN_EPS
    # --per_label
