import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from network_data import Network
from jpso_double import Swarm as Swarm_v1
from jpso_double_v2 import Swarm as Swarm_v2
from steinertree import Tree
import random

fp = os.path.join(os.path.dirname(__file__),'../../WusnNewModel/data/medium_data/uu-dem5_r40_1.in')

nw = Network(fp)

s = Swarm_v1(nw)

r = s.eval()

print(r)
