python3 pre_processing/patch_gen_hdf5.py \
        --ref_file ./examples/reference_LUAD.csv \
        --patch_path ./bulk_rna/LUAD/Patches_hdf5 \
        --mask_path ./bulk_rna/LUAD/Patches_hdf5 \
        --patch_size 256 \
        --max_patches_per_slide 4000

python3 pre_processing/compute_features_hdf5.py \
        --ref_file ./examples/reference_LUAD.csv \
        --patch_data_path ./bulk_rna/LUAD/Patches_hdf5 \
        --feature_path ./bulk_rna/LUAD/features \
        --max_patch_number 4000 \
        --feat_type resnet \
        --gpu_id 3

python3 pre_processing/kmean_features.py \
        --ref_file ./examples/reference_LUAD.csv  \
        --patch_data_path ./bulk_rna/LUAD/Patches_hdf5 \
        --feature_path ./bulk_rna/LUAD/features  \
        --num_clusters 100

python3 pre_processing/compute_features_hdf5.py \
        --ref_file ./examples/reference_LUAD.csv \
        --patch_data_path ./bulk_rna/LUAD/Patches_hdf5 \
        --feature_path ./bulk_rna/LUAD/uni_features \
        --max_patch_number 4000 \
        --feat_type uni \
        --gpu_id 3

python3 pre_processing/kmean_features.py \
        --ref_file ./examples/reference_LUAD.csv  \
        --patch_data_path ./bulk_rna/LUAD/Patches_hdf5 \
        --feature_path ./bulk_rna/LUAD/uni_features  \
        --num_clusters 100
