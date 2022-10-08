#!/bin/bash
cd ~/IKD-mmt
MSCOCO_Ckpt_dir=../checkpoints/mscoco17/en-de/inverseKD_res_m_l2
MSCOCO_Data_save=../data/mscoco17/en-de/inverseKD

# pre-train on MSCOCO, 4 gpus 50k equal to 200k
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py $MSCOCO_Data_save/pre \
#--distributed-no-spawn \
#--task translation_immt --arch transformer_immt_tiny --share-all-embeddings --dropout 0.3  \
#--optimizer adam --adam-betas 0.9,0.98 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
#--warmup-updates 2000 --lr 0.004 --min-lr 1e-09 --criterion label_smoothed_immt_cross_entropy --label-smoothing 0.1 \
#--pretrained-cnn resnet50 --loss1-coeff 1. --loss2-coeff 1. --dist-func l2 --grain model \
#--max-tokens 1024 --save-dir $MSCOCO_Ckpt_dir --log-format json --max-update 50000 --find-unused-parameters --no-double-encoder #--patience 10

# fine-tune on Multi30K, 4 gpus 15k equal to 60K, same save-dir
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py $MSCOCO_Data_save/fine \
--distributed-no-spawn \
--task translation_immt --arch transformer_immt_tiny --share-all-embeddings --dropout 0.3  \
--optimizer adam --adam-betas 0.9,0.98 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--warmup-updates 2000 --lr 0.004 --min-lr 1e-09 --criterion label_smoothed_immt_cross_entropy --label-smoothing 0.1 \
--pretrained-cnn resnet50 --loss1-coeff 1. --loss2-coeff 1. --dist-func l2 --grain model \
--max-tokens 1024 --save-dir $MSCOCO_Ckpt_dir --log-format json --max-update 65000 --find-unused-parameters --no-double-encoder #--patience 10
