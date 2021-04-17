#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python search_space.py --search normal
CUDA_VISIBLE_DEVICES=$1 python search_space.py --search ee --th_param 3.0
CUDA_VISIBLE_DEVICES=$1 python search_space.py --search ee --th_param 4.0
CUDA_VISIBLE_DEVICES=$1 python search_space.py --search ee --th_param 5.0



