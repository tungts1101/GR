import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from network_data import Network
from jpso_double import Swarm as Swarm_v1,Particle as P1
from jpso_double_v2 import Swarm as Swarm_v2,Particle as P2
from steinertree import Tree
import random

fp = os.path.join(os.path.dirname(__file__),'../../WusnNewModel/data/small_data/uu-dem3_r40_1.in')

nw = Network(fp)

# p1 = P1(nw)
# #print(p1)
# p2 = P2(nw)
# #print(p2)

# t = Tree(root=nw.sink,compulsory_nodes=nw.comp_nodes)
# t.decode(p1.layer_2,nw.distance,nw.trans_range)

# if not t.is_fisible(nw.distance,nw.trans_range):
    # print('NOt OK')
# else:
    # print("OK")

s_1 = Swarm_v1(nw)
s_2 = Swarm_v2(nw)

r_1 = s_1.eval()
r_2 = s_2.eval()

print(r_1)
print(r_2)
