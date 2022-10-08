#!/bin/bash
cd ~/IKD-mmt

MSCOCO_Data_save=../data/mscoco17/en-de/inverseKD
MSCOCO_Data_dir=../data/mscoco17/en-de
MSCOCO_Image_dir=../data/mscoco17/image_splits

Multi30k_Data_dir=../data/multi30k/en-de
Multi30k_Image_dir=../data/multi30k/image_splits

# cat mscoco2017 and multi30k datasets
#rm -rf $MSCOCO_Data_dir/train.lc.norm.tok.bpe.comb.en $MSCOCO_Data_dir/train.lc.norm.tok.bpe.comb.de
#cat $MSCOCO_Data_dir/train.lc.norm.tok.bpe.en $Multi30k_Data_dir/train.lc.norm.tok.bpe.en >> $MSCOCO_Data_dir/train.lc.norm.tok.bpe.comb.en
#cat $MSCOCO_Data_dir/train.lc.norm.tok.bpe.de $Multi30k_Data_dir/train.lc.norm.tok.bpe.de >> $MSCOCO_Data_dir/train.lc.norm.tok.bpe.comb.de

# create and preserve the concatenated dict
#python preprocess.py --source-lang en --target-lang de --trainpref $MSCOCO_Data_dir/train.lc.norm.tok.bpe.comb \
#--validpref $MSCOCO_Data_dir/val.lc.norm.tok.bpe --testpref $MSCOCO_Data_dir/val.lc.norm.tok.bpe \
#--destdir $MSCOCO_Data_save --joined-dictionary --workers 16
#rm -rf $MSCOCO_Data_save/*.idx $MSCOCO_Data_save/*.bin

# pre-train on MSCOCO
#python preprocess_immt.py --source-lang en --target-lang de --trainpref $MSCOCO_Data_dir/train.lc.norm.tok.bpe \
#--validpref $MSCOCO_Data_dir/val.lc.norm.tok.bpe --testpref $MSCOCO_Data_dir/val.lc.norm.tok.bpe \
#--srcdict $MSCOCO_Data_save/dict.en.txt --tgtdict $MSCOCO_Data_save/dict.de.txt --pretrained-cnn resnet50 --images-path ../data/mscoco17/all-images --train-fnames $MSCOCO_Image_dir/train.txt \
#--valid-fnames $MSCOCO_Image_dir/val.txt --destdir $MSCOCO_Data_save/pre --image-suffix image --workers 16

# fine-tune on Multi30K, shared dict with MSCOCO and new destdir
python preprocess_immt.py --source-lang en --target-lang de --trainpref $Multi30k_Data_dir/train.lc.norm.tok.bpe \
--validpref $Multi30k_Data_dir/val.lc.norm.tok.bpe --testpref $Multi30k_Data_dir/test_2016_flickr.lc.norm.tok.bpe,\
$Multi30k_Data_dir/test_2017_flickr.lc.norm.tok.bpe,$Multi30k_Data_dir/test_2017_mscoco.lc.norm.tok.bpe \
--srcdict $MSCOCO_Data_save/dict.en.txt --tgtdict $MSCOCO_Data_save/dict.de.txt \
--pretrained-cnn resnet50 --images-path ../data/multi30k/all-images --train-fnames $Multi30k_Image_dir/train.txt \
--valid-fnames $Multi30k_Image_dir/val.txt --destdir $MSCOCO_Data_save/fine --image-suffix image --workers 16
