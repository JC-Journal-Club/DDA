#!/bin/bash

data_path="/home/EGG_Data/SEED/ExtractedFeatures/"
session=("2")

for s in ${session[@]}; do
    echo "$data_path$s"
    for f in `ls $data_path$s`; do
        CUDA_VISIBLE_DEVICES=1 python3 train.py --session=$s --tar_file=$f
        # echo $s $f
    done
done