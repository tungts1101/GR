import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../WusnNewModel'))

import random
from steinertree import Tree
import matplotlib.pyplot as plt

time_slot = 1.5
Ee = 50*1e-9        # energy consumption of radio dissipation (nj/bit)
Efs = 100*1e-12        # energy consumption in the free space channel model (pj/bit/m^2)
Emp = 50*1e-12        # energy consumption in multi-path fading channel model (pj/bit/m^2)
Eda = 5*1e-9        # energy consumption for data aggregation (nj/bit)
k = 4000            # data sending and receiving (bit)
E = 50              # initial energy

class Network:
    def __init__(self,sink,sources,area,N,trans_range):    
        def random_deploy(N,area):
            (w,h) = area
            coord = [(i,j) for i in range(w+1) for j in range(h+1)]
            return random.sample(coord,N) 
        self.sink = sink
        self.sources = sources
        self.comp_nodes = [self.sink] + self.sources
        self.N = N
        self.relays = [node for node in range(self.N) if node is not sink and node not in sources]
        
        self.area = area
        self.trans_range = trans_range
        self.coord = random_deploy(self.N,self.area)

    def __cal_Etx(self,d):
        if d < self.trans_range:
            return k*Ee + k*Efs*d**2
        else:
            return k*Ee + k*Emp*d**2**2
    
    # calculate convergence time of tree
    # = sum(time slot in all descendants of all node in tree)
    def convergence_time(self,tree):
        return sum([len(tree.find_des(node))*time_slot for node in tree.tree])

    # calculate distance between 2 nodes
    def distance(self,n1,n2):
        x1,y1 = self.coord[n1]
        x2,y2 = self.coord[n2]
        return ((x1-x2)**2 + (y1-y2)**2)**(.5)
    
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
        return Etx + sum([len(tree.find_des(node))*(k*Ee+Eda)])
    
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
        
    def draw_vertices(self,vertices,shape):
        plt.plot([self.coord[i][0] for i in vertices],[self.coord[i][1] for i in vertices],shape)

    def draw_edge(self,v1,v2):
        x1,y1 = self.coord[v1]
        x2,y2 = self.coord[v2]
        plt.plot([x1,x2],[y1,y2],'-r')

    def show(self,tree):
        # draw vertices
        self.draw_vertices(self.sources,'gs')
        self.draw_vertices([self.sink],'b^')
        self.draw_vertices(self.relays,'ro')

        # draw edges
        for v1,v2 in tree.find_fringe():
            self.draw_edge(v1,v2)

        plt.axis([0,self.area[0],0,self.area[1]])
        plt.show()
