#!/bin/bash

### 设置该作业的作业名
#SBATCH --job-name=train

### 指定该作业需要1个节点数
#SBATCH --nodes=1

### 该作业需要8个CPU
#SBATCH --ntasks=4

### 申请1块GPU卡
#SBATCH --gres=gpu:1

### 作业脚本中的输出文件
#SBATCH --output=out/cl_train.%j.out

### 将作业提交到对应分区；
#SBATCH --partition=gpu

### 程序的执行命令
cd ~/IKD-mmt/experiments/en-de/trans/
sh train.sh