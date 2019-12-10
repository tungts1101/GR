import os
from steinertree import SteinerTree
from network import Network
from collections import defaultdict as dd
import random

fp = os.path.join(os.path.dirname(__file__),'../WusnNewModel/data/small_data/uu-dem8_r45_1.in')

nw = Network(fp)
st = SteinerTree(nw)
st.random_init()
# print(st.child)
# print('-----------------')
# print(st.parent)
# print('-----------------')
# print(st.leaves)
# print('-----------------')
print(nw.energy_consumption(st))
print(nw.communication_interference(st))
print(nw.convergence_time(st))
print(nw.network_lifetime(st))

def heuristics(st):
    def find_root(st, node):
        if st.parent[node] != None and st.parent[st.parent[node]] != None and nw.distance(st.parent[st.parent[node]], node) < nw.trans_range:
            cur_parent = st.parent[node]
            nex_parent = st.parent[st.parent[node]]

            st.child[cur_parent].remove(node)
            st.child[nex_parent].append(node)
            st.parent[node] = nex_parent

            find_root(st, node)
        else:
            return

    for node in st.terminals:
        find_root(st, node)
    
    st.trimming()
    # print(st.child)
    # print('----------------------')
    # print(st.parent)
    # print('----------------------')
    # print(st.leaves)
    print('----------------------')
    print(nw.energy_consumption(st))
    print(nw.communication_interference(st))
    print(nw.convergence_time(st))
    print(nw.network_lifetime(st))


heuristics(st)
