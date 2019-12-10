from GAwHeuristic.GA import *
from GAwHeuristic.heuristic import *
from common.input import *
from common.point import *
from common.dems_input import DemsInput
import time
import numpy as np 
from scipy import interpolate

def estimate(x, y, inp: DemsInput):
    id1 = int(x // inp.cellsize)
    id2 = int(y // inp.cellsize)
    xx = [id1 * inp.cellsize, (id1+1) * inp.cellsize]
    yy = [id2 * inp.cellsize, (id2+1) * inp.cellsize]
    z = [[inp.height[id1][id2], inp.height[id1][id2+1]], [inp.height[id1+1][id2], inp.height[id1+1][id2+1]]]
    plane = interpolate.interp2d(xx, yy, z)
    return plane(x, y)[0]

if __name__ == "__main__":
    inp = WusnInput.from_file('data/small_data/uu-dem1_r25_1.in')
    dem = DemsInput.from_file('data/dems_data/dem1.asc')
    dem.scale(41, 41, 5)
    print(estimate(8, 8, dem))
    # for sn in inp.relays:
    #     print(sn.z - estimate(sn.x, sn.y, dem))
    # z = np.zeros(shape=(1001, 1001),dtype=float)
    # for x in range(0, 1001):
    #     for y in range(0, 1001):
    #         if x % 25 == 0 and y % 25 == 0:
    #             z[x, y] = dem.height[int(x/25)][int(y/25)]
    #             # print(x, y)
    #         else:
    #             z[x, y] = estimate(x/25, y/25, dem)
    # print(z)
    # with open('dem1', 'w+') as f:
    #     for i in range(0, 1001):
    #         for j in range(0, 1001):
    #             f.write(str(z[i][j]) + ' ')
    #         f.write('\n')
    # inp = WusnInput.from_file("data/small_data/uu-dem1_r25_1.in")
    # sn_x = ''
    # sn_y = ''
    # sn_z = ''
    # rn_x = ''
    # rn_y = ''
    # rn_z = ''
    # for sn in inp.sensors:
    #     sn_x += str(sn.x) + ' '
    #     sn_y += str(sn.y) + ' '
    #     sn_z += str(sn.z) + ' '
    # for rn in inp.relays:
    #     rn_x += str(rn.x) + ' '
    #     rn_y += str(rn.y) + ' '
    #     rn_z += str(rn.z) + ' '
    # print(sn_x)
    # print(sn_y)
    # print(sn_z)
    # print(rn_x)
    # print(rn_y)
    # print(rn_z)
