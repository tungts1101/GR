import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../WusnNewModel'))

from steinertree import Tree
import random
from collections import defaultdict
from network_data import Network
from common.input import WusnInput as inp

fp = os.path.join(os.path.dirname(__file__),'../../WusnNewModel/data')

df = fp + '/small_data/'
f = df + 'ga-dem1_r25_1.in'

nw = inp.from_file(f)
# print('W = {},H = {}\n'.format(nw.W,nw.H))
# print('trans_range = {}\n'.format(nw.radius))
# print('Sources = {}\n'.format(nw.sensors))
# print('Relays = {}\n'.format(nw.relays))
# print('Sinks = {}\n'.format(nw.BS))

# if len(nw.sensors) == nw.num_of_sensors:
    # print('OK')

# if len(nw.relays) == nw.num_of_relays:
    # print('OK')

print(nw.BS.to_dict()['x'])
