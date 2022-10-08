#!/bin/bash

### 设置该作业的作业名
#SBATCH --job-name=pre

### 指定该作业需要1个节点数
#SBATCH --nodes=1

### 该作业需要8个CPU
#SBATCH --ntasks=8

### 申请1块GPU卡
#SBATCH --gres=gpu:1

### 作业脚本中的输出文件
#SBATCH --output=out/cl_pre.%j.out

### 将作业提交到对应分区；
#SBATCH --partition=gpu

### ????????????
cd ~/IKD-mmt/experiments/en-fr/dfmmt/
sh pre.sh