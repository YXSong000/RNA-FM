# #!/bin/bash

python3 src/main_tRNASformer_sequoia.py \
        --model_type vis \
        --ref_file examples/reference_LUAD.csv \
        --src_path ./rna_ckpts/sequoia-pub \
        --feature_path ./bulk_rna/LUAD/uni_features \
        --save_dir output \
        --cohort LUAD \
        --exp_name run_train_sequoia \
        --batch_size 16 \
        --k 5 \
        --train \
        --log sequoia \
        --save_on loss+corr \
        --stop_on loss+corr \
        --gpu_id 3 

python3 src/main_tRNASformer_sequoia_external.py \
        --model_type vis \
        --ref_file examples/reference_CPTAC_LUAD.csv \
        --src_path ./rna_ckpts/sequoia-pub \
        --feature_path ./bulk_rna/CPTAC_LUAD/uni_features \
        --checkpoint ./rna_ckpts/sequoia-pub/output/LUAD/run_train_sequoia/ \
        --save_dir output \
        --cohort CPTAC_LUAD \
        --exp_name run_train_noval_external \
        --batch_size 16 \
        --k 5 \
        --log sequoia \
        --save_on loss+corr \
        --stop_on loss+corr \
        --gpu_id 3
