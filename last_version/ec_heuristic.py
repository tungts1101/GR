import os
from steinertree import SteinerTree
from network import Network
from collections import defaultdict as dd
import random

fp = os.path.join(os.path.dirname(__file__),'../WusnNewModel/data/small_data/uu-dem8_r40_1.in')

nw = Network(fp)
# st = SteinerTree(nw)
# st.random_init()
# print(nw.energy_consumption(st))

# layer = [[st.root]]
# while(layer):
    # print(*layer)
    # layer = [st.child[c] for child in layer for c in child]

sns = set(node for node in nw.sources)
queue = [[nw.sink]]
neighbors = nw.adj_dict()
visited = set()
res = dd(lambda: [])


while queue:
    path = queue.pop(0)
    node = path[-1]
    if node in sns:
        res[node].append(path)

    if node not in visited:
        visited.add(node)

        #for neighbor in sorted(neighbors[node], key=lambda x:nw.distance(node,x)):
        for neighbor in neighbors[node]:
            if neighbor not in visited:
                new_path = list(path) + [neighbor]
                queue.append(new_path)

#print(res)

def contain_cycle(vp):
    father = dd(lambda: None)

    for path in vp:
        for i in range(len(path)-1,0,-1):
            x,y = path[i],path[i-1]
            if father[x] != None and father[x] != y:
                print(x,father[x],y)
                return True
            father[x] = y

    return False

vp = []
for sn in sns:
    # candidates = []
    # for path in res[sn]:
        # if len(path) == len(res[sn][0]):    # min hop
            # candidates.append(path)

    # def calculate_length(path):
        # return sum([nw.distance(x,y) for x,y in zip(path[0:-1],path[1:])])

    # candidates.sort(key=calculate_length)
    # #print("res 0 = {}, candidates = {}".format(res[sn][0], candidates[0]))

    # vp.append(candidates[0])
    vp.append(res[sn][0])


def build_tree_from_vp(vp):
    assert(not contain_cycle(vp))
    st = SteinerTree(nw)

    for path in vp:
        for i in range(0,len(path)-1):
            x,y = path[i],path[i+1]
            if y not in st.child[x]:
                st.child[x].append(y)
            st.parent[y] = x
        st.leaves.append(path[-1])

    return st

print(vp)
print('-------------------------')
st = build_tree_from_vp(vp)
assert(st.is_fisible())
print(nw.energy_consumption(st))

def cycle_reduction(vp):
    def remove_cycle(p1,p2):
        intersects = [e for e in p1 if e in p2]
        p_max = 0
        D_max = 0

        for p in intersects:
            D = abs(p1.index(p) - p2.index(p))
            if D >= D_max:
                p_max = p
                D_max = D

        if p_max != 0:
            i,j = p1.index(p_max),p2.index(p_max)
            p11,p12 = p1[:i],p1[i:]
            p21,p22 = p2[:j],p2[j:]

            if len(p21) <= len(p11):
                p1[:] = p21 + p12
            else:
                p2[:] = p11 + p22

    for i in range(len(vp)):
        for j in range(len(vp)):
            if i != j:
                remove_cycle(vp[i],vp[j])

sol = []

for node in sns:
    sol.append(random.choice(res[node]))

cycle_reduction(sol)
st = build_tree_from_vp(sol)
assert(st.is_fisible())
print(nw.energy_consumption(st))

def ec_heuristics(vp):
    for path in vp:
        if len(path) > 2:
            for i in range(len(path)-1,1,-1):
                j = i-2

                while nw.distance(i,j) < nw.trans_range:
                    j -= 1
                path[:] = path[:j+2] + path[i:]
                i = j+1

print(sol)
print('-----------------------------------')
ec_heuristics(sol)
print(sol)
st = build_tree_from_vp(sol)
print(nw.energy_consumption(st))
