#!/bin/bash
cd ~/IKD-mmt
Data_save=../data/multi30k/en-de/trans
Data_dir=../data/multi30k/en-de

# with bpe (en-de)
python preprocess.py --source-lang en --target-lang de --trainpref $Data_dir/train.lc.norm.tok.bpe \
--validpref $Data_dir/val.lc.norm.tok.bpe --testpref $Data_dir/test_2016_flickr.lc.norm.tok.bpe,\
$Data_dir/test_2017_flickr.lc.norm.tok.bpe,$Data_dir/test_2017_mscoco.lc.norm.tok.bpe \
--destdir $Data_save --joined-dictionary --workers 16