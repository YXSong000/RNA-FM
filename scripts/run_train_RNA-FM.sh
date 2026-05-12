# #!/bin/bash
# UNI features
python3 src/main_RNAFM.py \
        --model_type RNAFM \
        --ref_file examples/reference_LUAD.csv \
        --feature_path ./bulk_rna/LUAD/uni_features \
        --save_dir ./rna_ckpts/output \
        --cohort LUAD \
        --exp_name LUAD_uni_GOBP_ckpt \
        --depth 7 \
        --num_heads 8 \
        --batch_size 32 \
        --k 5 \
        --sample_every_epoch 30 \
        --patience 3 \
        --num_epochs 2000 \
        --num_sampling_steps 20 \
        --sampling_method euler \
        --cfg_scale 2.0 \
        --hidden_dim 256 \
        --lr 1e-4 \
        --train \
        --pathway_gene_indices_file ./examples/all_gene_indices_filtered.json \
        --gpu_id 2 

# RESNET features
python3 src/main_RNAFM.py \
        --model_type RNAFM \
        --ref_file examples/reference_LUAD.csv \
        --feature_path ./bulk_rna/LUAD/features \
        --save_dir ./rna_ckpts/output \
        --cohort LUAD \
        --exp_name LUAD_resnet_GOBP_ckpt \
        --depth 7 \
        --num_heads 8 \
        --batch_size 32 \
        --k 5 \
        --sample_every_epoch 30 \
        --patience 3 \
        --num_epochs 2000 \
        --num_sampling_steps 20 \
        --sampling_method euler \
        --cfg_scale 2.0 \
        --hidden_dim 256 \
        --lr 1e-4 \
        --train \
        --pathway_gene_indices_file ./examples/all_gene_indices_filtered.json \
        --gpu_id 2 

# MOLECULAR FUNCTION pathway
python3 src/main_RNAFM.py \
        --model_type RNAFM \
        --ref_file examples/reference_LUAD.csv \
        --feature_path ./bulk_rna/LUAD/uni_features \
        --save_dir ./rna_ckpts/output \
        --cohort LUAD \
        --exp_name LUAD_uni_GOBP_ckpt \
        --depth 7 \
        --num_heads 8 \
        --batch_size 32 \
        --k 5 \
        --sample_every_epoch 30 \
        --patience 3 \
        --num_epochs 2000 \
        --num_sampling_steps 20 \
        --sampling_method euler \
        --cfg_scale 2.0 \
        --hidden_dim 256 \
        --lr 1e-4 \
        --train \
        --pathway_gene_indices_file ./examples/all_gene_indices_molecular_function.json \
        --gpu_id 2 
