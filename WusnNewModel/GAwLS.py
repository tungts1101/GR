from GAwLS import *

if __name__ == '__main__':
    inp = WusnInput.from_file("data/small_data/dem1.in")
    start = time.time()
    abc = GA(inp)
    # print(inp.base_station)
    end = time.time()
    print(end-start)