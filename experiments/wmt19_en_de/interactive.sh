#!/bin/bash
cd ~/fairseq_wmt19ende

# --lenpen 0.6 作为新的翻译场景不需要长度惩罚
MODEL_DIR=../checkpoints/wmt19.en-de.joined-dict.ensemble
SAVE_DIR=../data/mscoco2017/raw

for f in train val; do
    cat $SAVE_DIR/$f.en | CUDA_VISIBLE_DEVICES=0 python interactive.py $MODEL_DIR/ \
                              --path $MODEL_DIR/model1.pt:$MODEL_DIR/model2.pt:$MODEL_DIR/model3.pt:$MODEL_DIR/model4.pt \
                              --beam 5 --batch-size 128 --buffer-size 128 --source-lang en --target-lang de \
                              --remove-bpe \
                              --tokenizer moses \
                              --bpe fastbpe --bpe-codes $MODEL_DIR/bpecodes >> $SAVE_DIR/generate-$f.de
                              
    # 提取最终翻译结果
    grep ^H $SAVE_DIR/generate-$f.de | sort -n -k 2 -t '-' | cut -f 3 >> $SAVE_DIR/$f.de
done