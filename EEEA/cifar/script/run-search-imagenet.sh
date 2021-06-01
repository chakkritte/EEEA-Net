#!/bin/bash


CUDA_VISIBLE_DEVICES=$1 python search_space.py --search normal --dataset imagenet --lmdb --init_channels_train 16 --init_channels 48 --increment 8 --parallel --learning_rate 0.5 --batch_size_train 1024 --batch_size_val 1024 --weight_decay 3e-4
#CUDA_VISIBLE_DEVICES=$1 python search_space.py --search normal --dataset imagenet --init_channels_train 16 --init_channels 48 --increment 8 --parallel --learning_rate 0.5 --batch_size_train 1024 --batch_size_val 1024
#CUDA_VISIBLE_DEVICES=$1 python search_space.py --search ee --th_param 3.0 --dataset imagenet --init_channels_train 16 --init_channels 48 --increment 8 --parallel --learning_rate 0.5 --batch_size_train 1024 --batch_size_val 1024
#CUDA_VISIBLE_DEVICES=$1 python search_space.py --search ee --th_param 4.0 --dataset imagenet --init_channels_train 16 --init_channels 48 --increment 8 --parallel --learning_rate 0.5 --batch_size_train 1024 --batch_size_val 1024
#CUDA_VISIBLE_DEVICES=$1 python search_space.py --search ee --th_param 5.0 --dataset imagenet --init_channels_train 16 --init_channels 48 --increment 8 --parallel --learning_rate 0.5 --batch_size_train 1024 --batch_size_val 1024


