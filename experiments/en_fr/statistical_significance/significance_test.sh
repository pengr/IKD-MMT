#!/bin/bash
cd /home/pengru/IKD-mmt

OUT_DIR=experiments/en_fr/statistical_significance/ss_result.txt
touch $OUT_DIR

# IKD-MMT vs MMT
./bootstrap-hypothesis-difference-significance.pl experiments/en_fr/statistical_significance/inverseKD_res_m_l2/test.txt experiments/en_fr/statistical_significance/mmt/test.txt \
experiments/en_fr/statistical_significance/org_test/test_2016_flickr.lc.norm.tok.fr
./bootstrap-hypothesis-difference-significance.pl experiments/en_fr/statistical_significance/inverseKD_res_m_l2/test1.txt experiments/en_fr/statistical_significance/mmt/test1.txt \
experiments/en_fr/statistical_significance/org_test/test_2017_flickr.lc.norm.tok.fr
./bootstrap-hypothesis-difference-significance.pl experiments/en_fr/statistical_significance/inverseKD_res_m_l2/test2.txt experiments/en_fr/statistical_significance/mmt/test2.txt \
experiments/en_fr/statistical_significance/org_test/test_2017_mscoco.lc.norm.tok.fr

# IKD-MMT vs Trans
./bootstrap-hypothesis-difference-significance.pl experiments/en_fr/statistical_significance/inverseKD_res_m_l2/test.txt experiments/en_fr/statistical_significance/trans/test.txt \
experiments/en_fr/statistical_significance/org_test/test_2016_flickr.lc.norm.tok.fr
./bootstrap-hypothesis-difference-significance.pl experiments/en_fr/statistical_significance/inverseKD_res_m_l2/test1.txt experiments/en_fr/statistical_significance/trans/test1.txt \
experiments/en_fr/statistical_significance/org_test/test_2017_flickr.lc.norm.tok.fr
./bootstrap-hypothesis-difference-significance.pl experiments/en_fr/statistical_significance/inverseKD_res_m_l2/test2.txt experiments/en_fr/statistical_significance/trans/test2.txt \
experiments/en_fr/statistical_significance/org_test/test_2017_mscoco.lc.norm.tok.fr
