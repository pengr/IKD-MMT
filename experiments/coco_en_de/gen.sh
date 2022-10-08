#!/bin/bash
cd ~/IKD-mmt
MSCOCO_Ckpt_dir=../checkpoints/mscoco17/en-de/inverseKD_res_m_l2
MSCOCO_Data_save=../data/mscoco17/en-de/inverseKD
Multi30k_Data_dir=../data/multi30k/en-de
Meteor_dir=../meteor-1.5/meteor-1.5.jar

for((i=99;i>=90;i--)); do
  if [ ! -f "$MSCOCO_Ckpt_dir/checkpoint_last10_avg.pt" ]; then
    python scripts/average_checkpoints.py \
    --inputs $MSCOCO_Ckpt_dir \
    --output $MSCOCO_Ckpt_dir/checkpoint_last10_avg.pt \
    --num-epoch-checkpoints  10 \
    --checkpoint-upper-bound $i
  fi
  
  for f in valid test test1 test2; do
      CUDA_VISIBLE_DEVICES=0 python -W ignore generate.py $MSCOCO_Data_save/fine \
      --task translation_immt --criterion label_smoothed_immt_cross_entropy \
      --path $MSCOCO_Ckpt_dir/checkpoint_last10_avg.pt --gen-subset $f --beam 5 --batch-size 128 \
      --remove-bpe --results-path $MSCOCO_Ckpt_dir
  done
  
  echo $i >> $MSCOCO_Ckpt_dir/result.log
  
  for f in valid test test1 test2; do
      rm -f $MSCOCO_Ckpt_dir/$f.txt
      grep ^H $MSCOCO_Ckpt_dir/generate-$f.txt | sort -n -k 2 -t '-' | cut -f 3 >> $MSCOCO_Ckpt_dir/$f.txt
      cat $MSCOCO_Ckpt_dir/generate-$f.txt | tail -1
      
      if [ $f = valid ]; then
          java -Xmx2G -jar $Meteor_dir $MSCOCO_Ckpt_dir/$f.txt $Multi30k_Data_dir/val.lc.norm.tok.de -l de -norm -q -vOut | tail -1
      elif [ $f = test ]; then
          java -Xmx2G -jar $Meteor_dir $MSCOCO_Ckpt_dir/$f.txt $Multi30k_Data_dir/test_2016_flickr.lc.norm.tok.de -l de -norm -q -vOut | tail -1
      elif [ $f = test1 ]; then
          java -Xmx2G -jar $Meteor_dir $MSCOCO_Ckpt_dir/$f.txt $Multi30k_Data_dir/test_2017_flickr.lc.norm.tok.de -l de -norm -q -vOut | tail -1
      else
          java -Xmx2G -jar $Meteor_dir $MSCOCO_Ckpt_dir/$f.txt $Multi30k_Data_dir/test_2017_mscoco.lc.norm.tok.de -l de -norm -q -vOut | tail -1
      fi
  done >> $MSCOCO_Ckpt_dir/result.log
  
  rm -rf $MSCOCO_Ckpt_dir/checkpoint_last10_avg.pt
  rm -rf $MSCOCO_Ckpt_dir/*.txt
done
