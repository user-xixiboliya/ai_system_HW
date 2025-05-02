#!/usr/bin/env bash

# 使用NCCL后端（默认）
python main.py --backend nccl --num_gpus 4

# 使用Gloo后端
# python main.py --backend gloo --num_gpus 1

# MPI
# python main.py --backend mpi --num_gpus 1

# 使用2个GPU训练
CUDA_VISIBLE_DEVICES=0,1 python main.py --batch_size 128