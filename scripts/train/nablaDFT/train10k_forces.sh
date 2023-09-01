#!/bin/bash

# Loading the required module

python main_nablaDFT_forces.py \
    --output-dir 'models/nablaDFT/equiformer/se_l2/target@0_forces/' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \
    --data-path '/mnt/2tb/khrabrov/nablaDFT_pyg/train_10k' \
    --batch-size 32 \
    --eval-batch-size 16 \
    --radius 5.0 \
    --num-basis 32 \
    --drop-path 0.0 \
    --weight-decay 1e-6 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --split train_10k \
    --energy-weight 1 \
    --force-weight 80 \
    --train-size 44753 \
    --val-size 4972 
