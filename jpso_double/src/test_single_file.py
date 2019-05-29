import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from network_data import Network
from ga import Swarm as Swarm_v2,Particle as P2
from pso import Swarm as Swarm_v1
from steinertree import Tree
import random
from multiprocessing import Process,Value

fp = os.path.join(os.path.dirname(__file__),'../../WusnNewModel/data/medium_data/uu-dem3_r40_1.in')

nw = Network(fp)
s = Swarm_v1(nw)
r = s.eval()

print("Value = {:.2f} Time = {:.2f}".format(r['value'],r['time']))
