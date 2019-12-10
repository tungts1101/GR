#!/usr/bin/env bash

DATASET=${1-"small"}
PROCS=2
ALPHA=0.5

python3 exact.py -p $PROCS --alpha $ALPHA -i 1\
    -o data/${DATASET}_data/approx\
    --lax data/${DATASET}_data/*.in #> data/${DATASET}_data/BOUND1