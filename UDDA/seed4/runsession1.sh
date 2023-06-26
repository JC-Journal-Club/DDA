#!/bin/bash

data_path="/data2/EEG_data/SEED4/eeg_feature_smooth/"
session=("1")

for s in ${session[@]}; do
    echo "$data_path$s"
    for f in `ls $data_path$s`; do
        CUDA_VISIBLE_DEVICES=2 python3 train.py --session=$s --tar_file=$f
        # echo $s $f
    done
done