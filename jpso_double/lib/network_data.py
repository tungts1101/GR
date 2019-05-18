import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../WusnNewModel'))

import random
from steinertree import Tree
from common.input import WusnConstants as cf
from common.input import WusnInput as inp

time_slot = 1.5
Erx = cf.e_rx
Efs = cf.e_fs
Eda = cf.e_da
Emp = cf.e_mp
k = cf.k_bit
E = 50

class Network:
    def __init__(self,filename):
        def __coord(point):
            return (point.to_dict()['x'],point.to_dict()['y'],point.to_dict()['z'])

        nw = inp.from_file(filename)
        self.sink = 0
        self.sources = [i for i in range(1,nw.num_of_sensors+1)]
        self.comp_nodes = [self.sink] + self.sources
        self.relays = [i for i in range(nw.num_of_sensors+1,nw.num_of_sensors+nw.num_of_relays+1)]
        self.N = nw.num_of_sensors + nw.num_of_relays + 1
        self.area = (nw.W,nw.H)
        self.trans_range = 2*nw.radius

        self.coord = [__coord(p) for p in [nw.BS] + nw.sensors + nw.relays]

    def __cal_Etx(self,d):
        if d < self.trans_range/2:
            return Erx + k*Efs*d**2
        else:
            return Erx + k*Emp*d**2**2
    
    # calculate convergence time of tree
    # = sum(time slot in all descendants of all node in tree)
    def convergence_time(self,tree):
        return sum([len(tree.find_des(node))*time_slot for node in tree.tree])

    # calculate distance between 2 nodes
    def distance(self,n1,n2):
        x1,y1,z1 = self.coord[n1]
        x2,y2,z2 = self.coord[n2]
        return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(.5)
    
    # calculate link interference between 2 nodes
    # = total nodes can be affected in between
    def __link_interference(self,n1,n2):
        d = self.distance(n1,n2)

        return len([node for node in range(self.N) if (self.distance(node,n1) < d or self.distance(node,n2) < d) and (node is not n1 and node is not n2)])
    
    # calculate communication interference of tree
    # simply equal max of link interference
    def communication_interference(self,tree):
        return max([self.__link_interference(x,y) for x,y in tree.find_fringe()])
    
    ######
    # calculate energy consumption for one node
    # energy consumption = energy to receive + energy to transmit
    ######
    def __node_consumption(self,tree,node):
        parent = tree.find_parent(node)
        # energy to transmit to its parent
        Etx = 0 if parent is None else self.__cal_Etx(self.distance(node,parent))
        return Etx + sum([len(tree.find_des(node))*(Erx+Eda)])
    
    # calculate energy consumption of tree
    def energy_consumption(self,tree):
        return sum([self.__node_consumption(tree,node) for node in tree.tree])
    
    # calculate network life time of tree
    def network_lifetime(self,tree):
        return 1/min([E-self.__node_consumption(tree,node) for node in tree.tree])
    
    def find_edges(self):
        edges = []
        for i in range(self.N):
            for j in range(i+1,self.N):
                if self.distance(i,j) <= self.trans_range:
                    edges.append((i,j))

        return edges
        
    # def draw_vertices(self,vertices,shape):
        # plt.plot([self.coord[i][0] for i in vertices],[self.coord[i][1] for i in vertices],shape)

    # def draw_edge(self,v1,v2):
        # x1,y1 = self.coord[v1]
        # x2,y2 = self.coord[v2]
        # plt.plot([x1,x2],[y1,y2],'-r')

    # def show(self,tree):
        # # draw vertices
        # self.draw_vertices(self.sources,'gs')
        # self.draw_vertices([self.sink],'b^')
        # self.draw_vertices(self.relays,'ro')

        # # draw edges
        # for v1,v2 in tree.find_fringe():
            # self.draw_edge(v1,v2)

        # plt.axis([0,self.area[0],0,self.area[1]])
        # plt.show()
