#!/bin/bash
set -e

result_dir=$1

CMD_NIH="python Inference_NIH14.py --dir=True"
CMD_OPENI="python Inference_OPENI.py"
CMD_CXP="python Inference_CheXpert.py"
CMD_PAD="python Inference_PadChest.py"

if [ -n "$result_dir" ]; then
    CMD_NIH="$CMD_NIH --result_dir $result_dir"
    CMD_OPENI="$CMD_OPENI --result_dir $result_dir"
    CMD_CXP="$CMD_CXP --result_dir $result_dir"
    CMD_PAD="$CMD_PAD --result_dir $result_dir"
fi

echo "Inference NIH14 Dataset"
eval $CMD_NIH

echo "Inference OPENI Dataset"
eval $CMD_OPENI

echo "Inference CheXpert Dataset"
eval $CMD_CXP

echo "Inference PadChest Dataset"
eval $CMD_PAD
