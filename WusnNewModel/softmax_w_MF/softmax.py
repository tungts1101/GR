import time, os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from common.input import *

def max_num_of_sensor_to_relay(inp: WusnInput):
    max_num_of_sensor_to_rn = {}
    R = inp.radius
    for rn in inp.relays:
        max_num_of_sensor_to_rn[rn] = 0
        for sn in inp.sensors:
            if distance(sn, rn) <= 2*R:
                max_num_of_sensor_to_rn[rn] += 1
    return max_num_of_sensor_to_rn


def softmax(inp: WusnInput):
    max_num = max_num_of_sensor_to_relay(inp)
    weight = {}
    for rn in inp.relays:
        weight[rn] = inp.static_relay_loss[rn] + max_num[rn]*inp.dynamic_relay_loss[rn]
    rns = []
    for i in range(inp.num_of_relays):
        rns.append(i)
    for i in range(inp.num_of_relays):
        for j in range(i+1, inp.num_of_relays):
            if weight[inp.relays[i]] > weight[inp.relays[j]]:
                rns[i], rns[j] = rns[j], rns[i]
    return max_num, rns