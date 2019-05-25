import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from network_data import Network
from jpso_double import Swarm as Swarm_v1,Particle as P1
from jpso_double_v2 import Swarm as Swarm_v2,Particle as P2
from steinertree import Tree
import random
from multiprocessing import Process,Value

fp = os.path.join(os.path.dirname(__file__),'../../WusnNewModel/data/small_data/uu-dem3_r40_1.in')

nw = Network(fp)

v = Value('d',0.0)
t = Value('d',0.0)

def handle_file():
    s = Swarm_v2(nw)
    r = s.eval('sb')
    print("{:.2f} {:.2f}".format(r['value'],r['time']))
    
    v.value += r['value']
    t.value += r['time']

print('select best')
all_proc = [Process(target=handle_file,args=()) for i in range(30)]

for p in all_proc:
    p.start()

for p in all_proc:
    p.join()

print("Value = {:.2f} Time = {:.2f}".format(v.value/30,t.value/30))
