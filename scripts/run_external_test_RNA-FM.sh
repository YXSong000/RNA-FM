#!/bin/bash

python3 src/main_RNAFM_external_test.py \
        --model_type RNAFM \
        --ref_file examples/reference_CPTAC_LUAD.csv \
        --feature_path ./bulk_rna/CPTAC_LUAD/uni_features \
        --save_dir ./rna_ckpts/output \
        --cohort CPTAC_LUAD \
        --ckpt_path ./rna_ckpts/output/LUAD/LUAD_uni_GOBP_ckpt \
        --exp_name LUAD_uni_GOBP_ckpt \
        --depth 7 \
        --num_heads 8 \
        --batch_size 32 \
        --k 5 \
        --num_sampling_steps 20 \
        --sampling_method euler \
        --cfg_scale 2.0 \
        --hidden_dim 256 \
        --pathway_gene_indices_file ./examples/all_gene_indices_filtered.json \
        --gpu_id 3
