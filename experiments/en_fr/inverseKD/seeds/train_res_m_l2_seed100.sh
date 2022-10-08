#!/bin/bash
cd ~/IKD-mmt
Ckpt_dir=../checkpoints/multi30k/en-fr/inverseKD_res_m_l2_seed100
Data_save=../data/multi30k/en-fr/inverseKD

CUDA_VISIBLE_DEVICES=3 python train.py $Data_save \
--task translation_immt --arch transformer_immt_tiny --share-all-embeddings --dropout 0.3  \
--optimizer adam --adam-betas 0.9,0.98 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--warmup-updates 2000 --lr 0.004 --min-lr 1e-09 --criterion label_smoothed_immt_cross_entropy --label-smoothing 0.1 \
--pretrained-cnn resnet50 --loss1-coeff 1. --loss2-coeff 1. --dist-func l2 --grain model \
--max-tokens 1024 --save-dir $Ckpt_dir --log-format json --max-update 50000 --seed 100 --find-unused-parameters --no-double-encoder --patience 10