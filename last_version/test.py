import os

from dle import DLE
from steinertree import SteinerTree
from network import Network
from pso import Swarm as spso
from ga import Swarm as sga
from gaea import Swarm as sgaea
import random

fp = os.path.join(os.path.dirname(__file__),'../WusnNewModel/data/small_data/uu-dem8_r40_1.in')

nw = Network(fp)
swarm = spso(nw)
ret = swarm.eval()
print(ret)
