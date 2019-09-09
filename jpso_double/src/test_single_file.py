import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from network_data import Network
from im_moea import Swarm as Swarm_im
from nsga_ii import Swarm as Swarm_nsga
from ga import Swarm as Swarm_ga
from pso import Swarm as Swarm_pso

fp = os.path.join(os.path.dirname(__file__),'../../WusnNewModel/data/small_data/uu-dem3_r40_1.in')

nw = Network(fp)
s = Swarm_nsga(nw)
r = s.eval()
print(r["error"],r["running_time"],r["generations"])

s = Swarm_ga(nw)
r = s.eval()
print(r["error"],r["running_time"],r["generations"])

s = Swarm_pso(nw)
r = s.eval()
print(r["error"],r["running_time"],r["generations"])
