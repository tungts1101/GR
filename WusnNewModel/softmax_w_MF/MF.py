import os, sys
from ortools.graph import pywrapgraph
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from common.input import *

def generate_connect_matrix(inp: WusnInput, max_num, relays):
    R = inp.radius
    connect = {}
    for rn in relays:
        count = 0
        for sn in inp.sensors:
            if distance(sn, rn) <= 2*R:
                count += 1
                connect[(sn, rn)] = 1
        if count == 0:
            return False
    return connect

def max_flow(inp: WusnInput, max_num, rns):
    start_nodes = []
    end_nodes = []
    capacities = []
    cost = []
    supply = inp.num_of_relays + inp.num_of_sensors + len(rns) + 10000
    demand = inp.num_of_relays + inp.num_of_sensors + len(rns) + 20000
    
    for i in range(inp.num_of_sensors):
        start_nodes.append(supply)
        end_nodes.append(i)
        capacities.append(1)
        cost.append(0)
    
    for i in range(inp.num_of_sensors):
        for j in rns:
            if distance(inp.relays[j], inp.sensors[i]) <= 2*inp.radius:
                start_nodes.append(i)
                end_nodes.append(j+inp.num_of_sensors) 
                capacities.append(1)
                # cost.append()

    for i in rns:
        start_nodes.append(i+inp.num_of_sensors)
        end_nodes.append(i+inp.num_of_sensors+len(rns))
        capacities.append(max_num[inp.relays[i]])

        start_nodes.append(i+inp.num_of_sensors+len(rns))
        end_nodes.append(demand)
        capacities.append(max_num[inp.relays[i]])
        # cost.append(inp.static_relay_loss(inp.relays[i]))
    
    for i in range(len(start_nodes)):
        print(start_nodes[i], end_nodes[i], capacities[i])

    mf = pywrapgraph.SimpleMaxFlow()

    for i in range(0, len(start_nodes)):
        mf.AddArcWithCapacity(start_nodes[i], end_nodes[i], capacities[i])

    if mf.Solve(supply, demand) == mf.OPTIMAL:
        print('Max flow:', mf.OptimalFlow())
        print('')
        print('  Arc    Flow / Capacity')
        for i in range(mf.NumArcs()):
            if mf.Flow(i) != 0:
                print('%1s -> %1s   %3s  / %3s' % (
                    mf.Tail(i),
                    mf.Head(i),
                    mf.Flow(i),
                    mf.Capacity(i)))
        print('Source side min-cut:', mf.GetSourceSideMinCut())
        print('Sink side min-cut:', mf.GetSinkSideMinCut())
    else:
        print('There was an issue with the max flow input.')