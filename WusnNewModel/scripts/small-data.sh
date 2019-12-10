#!/usr/bin/env bash
rm data/small_data/*.in

python3 scripts/datagen.py -o data/small_data\
    -W 200 -H 200 --depth 1 --height 10\
     --rows 41 --cols 41 --num-sensor 40 --num-relay 40\
     --csize 5 --radius 25,40\
     data/dems_data/*.asc
