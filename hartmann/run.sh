#!/bin/bash

exp=${1:-"romo"}
device=${2:-0}
runs=${3:-10}

echo "Run ${exp} ${runs} times on cuda:${device}"

for i in `seq ${runs}`
do
CUDA_VISIBLE_DEVICES=${device} python main.py -c ${exp}
done