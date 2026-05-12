#!/bin/bash

python3 src/main_tRNASformer_sequoia.py \
        --model_type vit \
        --ref_file examples/reference_LUAD.csv \
        --src_path ./rna_ckpts/tRNAsformer \
        --feature_path ./bulk_rna/LUAD/uni_features \
        --save_dir output \
        --cohort LUAD \
        --exp_name run_train_tRNAsformer \
        --batch_size 16 \
        --k 5 \
        --train \
        --log tRNAsformer \
        --save_on loss+corr \
        --stop_on loss+corr \
        --gpu_id 3 

python3 src/main_tRNASformer_sequoia_external.py \
        --model_type vit \
        --ref_file examples/reference_CPTAC_LUAD.csv \
        --src_path ./rna_ckpts/tRNAsformer \
        --feature_path ./bulk_rna/CPTAC_LUAD/uni_features \
        --checkpoint ./rna_ckpts/tRNAsformer/output/LUAD/run_train_tRNAsformer/ \
        --save_dir output \
        --cohort CPTAC_LUAD \
        --exp_name run_train_tRNAsformer_external \
        --batch_size 16 \
        --k 5 \
        --log tRNAsformer \
        --save_on loss+corr \
        --stop_on loss+corr \
        --gpu_id 3
