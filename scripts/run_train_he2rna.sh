#!/bin/bash

python3 src/main_he2rna.py \
        --path_csv examples/reference_LUAD.csv \
        --feature_path ./bulk_rna/LUAD/uni_features \
        --destfolder ./rna_ckpts/he2rna \
        --subfolder he2rna \
        --exp_name LUAD \
        --lr 1e-3 \
        --cohort LUAD \
        --gpu_id 3 
