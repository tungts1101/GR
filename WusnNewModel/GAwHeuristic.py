from GAwHeuristic.GA import *
from GAwHeuristic.heuristic import *
from common.input import *
from common.point import *
import time

alpha = 0.5

if __name__ == "__main__":
    # inp = WusnInput.from_file("small_data/dem3.in")
    # start = time.time()
    # indi = random_init_individual(inp.num_of_relays)
    # GA(inp)
    # a, b, c = heuristic(inp, indi)
    # print(a, b, c)
    # end = time.time()
    # print(end-start)

    for i in range (10, 20):
        res_file_name = "./small_data_result/small_data_result_"+str(i)+".txt"
        f = open(res_file_name, "w")
        file_list = os.listdir("data/small_data")
        for file_name in file_list:
            if file_name == "BOUND":
                continue
            print(str(file_name))
            inp = WusnInput.from_file("data/small_data/" + str(file_name))
            start = time.time()
            res = GA(inp)[1]
            num_used_relays = len(res.used_relays)
            consumption = res.max_consumption()
            end = time.time()
            print(end-start)

            f.write(str(file_name) + "|")
            f.write(str(num_used_relays) + "|")
            f.write(str(consumption) + "|")
            f.write(str(res.loss(alpha)) + "|")
            f.write(str(end-start) + "s" + "\n")
            print()
        f.close()


    # f = open("medium_data_result.txt", "w+")
    # file_list = os.listdir("medium_data")
    # for file_name in file_list:
    #     print(str(file_name))
    #     f.write(file_name + "\n")
    #     inp = WusnInput.from_file("medium_data/" + str(file_name))
    #     start = time.time()
    #     res = GA(inp)
    #     end = time.time()
    #     print(end-start)
    #     f.write(str(res) + "\n")
    #     f.write(str(end-start) + "s" + "\n")
    #     print()
    # f.close()
