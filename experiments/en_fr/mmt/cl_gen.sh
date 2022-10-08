#!/bin/bash

### ���ø���ҵ����ҵ��
#SBATCH --job-name=gen

### ָ������ҵ��Ҫ1���ڵ���
#SBATCH --nodes=1

### ����ҵ��Ҫ8��CPU
#SBATCH --ntasks=4

### ����1��GPU��
#SBATCH --gres=gpu:1

### ��ҵ�ű��е�����ļ�
#SBATCH --output=out/cl_gen.%j.out

### ����ҵ�ύ����Ӧ������
#SBATCH --partition=gpu

### �����ִ������
cd ~/IKD-mmt/experiments/en-fr/mmt/
sh gen.sh