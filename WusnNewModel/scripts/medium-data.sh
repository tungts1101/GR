#!/usr/bin/env bash
rm data/medium_data/*.in

python3 scripts/datagen.py -o data/medium_data\
    -W 500 -H 500 --depth 1 --height 10\
     --rows 101 --cols 101 --num-sensor 100 --num-relay 100\
     --csize 5 --radius 25,40\
     data/dems_data/*.asc
