import os
from common.input import *
from common.point import *

def adj(node, nodes, radius):
    res = []
    for n in nodes:
        if distance(n, node) <= 2*radius:
            res.append(n)
    return res

# def fix(f):
    # inp = WusnInput.from_file(f)
    # nodes = [inp.BS] + inp.sensors + inp.relays
    # radius = inp.radius
    # visited = set()

    # def connected():
        # q = [inp.BS]
        # visited = set()

        # while q:
            # node = q.pop(0)
            # if node not in visited:
                # visited.add(node)

                # for n in adj(node, nodes, radius):
                    # if n not in visited:
                        # q.append(n)
        # return all(node in visited for node in inp.sensors + [inp.BS])
    
    # while not connected():
        # for node in inp.sensors + [inp.BS]:
            # if node not in visited:
                # print(node)
                # print(adj(node, nodes, radius))
                # print('=============')

                # node.x = node.x - radius/5 if node.x > inp.BS.x else node.x + radius/5
                # node.y = node.y - radius/5 if node.y > inp.BS.y else node.y + radius/5
                # node.z = node.z - radius/5 if node.z > inp.BS.z else node.z + radius/5

# fix('data/medium_data/uu-dem4_r30_1.in')

def check_relays(f):
    inp = WusnInput.from_file(f)

    for rn in inp.relays:
        d = distance(inp.BS, rn)
        if d > 2*inp.radius:
            print(rn)
            print(d)
            print('------------')
        
#check_relays('data/medium_data/uu-dem4_r30_1.in')

def is_connected(f):
    inp = WusnInput.from_file(f)
    nodes = [inp.BS] + inp.sensors + inp.relays
    radius = inp.radius

    q = [inp.BS]
    visited = set()

    while q:
        node = q.pop(0)
        if node not in visited:
            visited.add(node)

            for n in adj(node, nodes, radius):
                q.append(n)

    return all(node in visited for node in inp.sensors + [inp.BS])

i = 0
for f in os.listdir('data/small_data'):
    if f[-2:] == 'in':
        if not is_connected('data/small_data/' + f):
            i += 1
            print('small_data/' + f)

for f in os.listdir('data/medium_data'):
    if f[-2:] == 'in':
        if not is_connected('data/medium_data/' + f):
            i += 1
            print('medium_data/' + f)

print(i)
