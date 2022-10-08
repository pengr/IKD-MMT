#!/bin/bash

### ����ҵ����ҵ��
#SBATCH --job-name=train_res_m_kl

### ����ҵ��Ҫ1���ڵ�
#SBATCH --nodes=1

### ����ҵ��Ҫ1��CPU
#SBATCH --ntasks=4

### ����1��GPU��
#SBATCH --gres=gpu:1

### ��ҵ�ű��е�����ļ�
#SBATCH --output=out/cl_train_res_m_kl.%j.out

### ����ҵ�ύ����Ӧ������
#SBATCH --partition=gpu

### �����ִ������
cd ~/IKD-mmt/experiments/en-de/inverseKD/
sh train_res_m_kl.sh