#small
#gaussian
python scripts/datagen.py -o data/small_data -W 200 -H 200 --depth 1 --height 10 --rows 41 --cols 41 --num-sensor 40 --num-relay 40 --csize 5 --radius 25,30,45,50 --prefix no- --distribution gaussian data/dems_data/*.asc
#gamma
python scripts/datagen.py -o data/small_data -W 200 -H 200 --depth 1 --height 10 --rows 41 --cols 41 --num-sensor 40 --num-relay 40 --csize 5 --radius 25,30,45,50 --prefix ga- --distribution gamma data/dems_data/*.asc
#uniform
python scripts/datagen.py -o data/small_data -W 200 -H 200 --depth 1 --height 10 --rows 41 --cols 41 --num-sensor 40 --num-relay 40 --csize 5 --radius 25,30,45,50 --prefix uu- --distribution uniform data/dems_data/*.asc

#medium
#gaussian 
python scripts/datagen.py -o data/medium_data -W 500 -H 500 --depth 1 --height 10 --rows 101 --cols 101 --num-sensor 100 --num-relay 100 --csize 5 --radius 25,30,45,50 --prefix no- --distribution gaussian data/dems_data/*.asc
#gamma
python scripts/datagen.py -o data/medium_data -W 500 -H 500 --depth 1 --height 10 --rows 101 --cols 101 --num-sensor 100 --num-relay 100 --csize 5 --radius 25,30,45,50 --prefix ga- --distribution gamma data/dems_data/*.asc
#uniform
python scripts/datagen.py -o data/medium_data -W 500 -H 500 --depth 1 --height 10 --rows 101 --cols 101 --num-sensor 100 --num-relay 100 --csize 5 --radius 25,30,45,50 --prefix uu- --distribution uniform data/dems_data/*.asc