#!/bin/bash

exp=${1:-"romo"}
env=${2:-"hopper"}
device=${3:-0}
runs=${4:-5}

echo "Run ${exp}-${env} ${runs} times on cuda:${device}"

for i in `seq ${runs}`
do
CUDA_VISIBLE_DEVICES=${device} python main.py -c ${exp}-${env}
done