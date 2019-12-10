from softmax_w_MF.softmax import *
from softmax_w_MF.MF import *
from common.input import *
from common.point import *
import time

if __name__ == "__main__":
    inp = WusnInput.from_file("small_data/dem1_0.in")
    num, rns = softmax(inp)    # return the list in asceding order
    max_flow(inp, num, rns)