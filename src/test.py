import os

from network import Network
from ga import Swarm as sga
from pso import Swarm as spso
from nsga import Swarm as snsga

fp = os.path.join(os.path.dirname(__file__), '../WusnNewModel/data/small_data/ga-dem1_r30_1.in')
nw = Network(fp)

s = sga(nw, swarm_size=10, generation=10)
r = s.eval()

s = spso(nw, swarm_size=10, generation=10)
r = s.eval()

s = snsga(nw, swarm_size=20, generation=10)
r = s.eval()
