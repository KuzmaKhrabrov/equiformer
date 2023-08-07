#!/bin/bash

# Loading the required module

python main_nablaDFT.py \
    --output-dir 'models/qm9/equiformer/se_l2/target@0/' \
    --model-name 'dot_product_attention_transformer_exp_l2_nablaDFTwoforces' \
    --input-irreps '64x0e' \
    --data-path '/mnt/2tb/khrabrov/nablaDFT_pyg_v2/' \
    --batch-size 128 \
    --radius 5.0 \
    --num-basis 32 \
    --drop-path 0.0 \
    --weight-decay 1e-6 \
    --lr 5e-4 \
    --min-lr 1e-6 \

