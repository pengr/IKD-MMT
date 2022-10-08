#!/bin/bash
cd ~/IKD-mmt
Data_save=../data/multi30k/en-de/inverseKD
Data_dir=../data/multi30k/en-de
Image_dir=../data/multi30k/image_splits

# with bpe (en-de) no --test-fnames
python preprocess_immt.py --source-lang en --target-lang de --trainpref $Data_dir/train.lc.norm.tok.bpe \
--validpref $Data_dir/val.lc.norm.tok.bpe --testpref $Data_dir/test_2016_flickr.lc.norm.tok.bpe,\
$Data_dir/test_2017_flickr.lc.norm.tok.bpe,$Data_dir/test_2017_mscoco.lc.norm.tok.bpe \
--pretrained-cnn resnet50 --images-path ../data/multi30k/all-images --train-fnames $Image_dir/train.txt \
--valid-fnames $Image_dir/val.txt --destdir $Data_save --image-suffix image --joined-dictionary --workers 16