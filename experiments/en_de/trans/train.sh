#!/bin/bash
cd ~/IKD-mmt
Ckpt_dir=../checkpoints/multi30k/en-de/trans
Data_save=../data/multi30k/en-de/trans

CUDA_VISIBLE_DEVICES=1 python train.py $Data_save \
--task translation --arch transformer_tiny --share-all-embeddings --dropout 0.3  \
--optimizer adam --adam-betas 0.9,0.98 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--warmup-updates 2000 --lr 0.004 --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 1024 --save-dir $Ckpt_dir --log-format json --max-update 40000 --find-unused-parameters --patience 10