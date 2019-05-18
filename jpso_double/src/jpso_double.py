import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

#from network import Network
from steinertree import Tree
import random
from collections import defaultdict, Counter
from functools import reduce
import time

class Particle:
    def __init__(self,network):
        self.network = network
        self.layer_2 = self.__gen_l2()
        self.layer_1 = self.__gen_l1_from_l2()
        self.p_layer_1 = []
        self.p_layer_2 = []
    
    def __gen_l2(self):
        # create a fisible solution
        t = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
        # find all eligible edges
        edges = self.network.find_edges()
        # shuffle edges to make our swarm more diversified
        random.shuffle(edges)
        t.kruskal(edges,self.network.N)
        if not t.is_fisible(self.network.distance,self.network.trans_range):
            print('Fail when create a tree in partice')
            exit(0)

        l2_len = 2*(self.network.N - 1)
        return t.encode(l2_len)
    
    def __gen_l1_from_l2(self):
        l1 = []
        for i in range(self.network.N):
            if i not in self.network.comp_nodes:
                if i in set(self.layer_2):
                    l1.append(1)
                else:
                    l1.append(0)
        
        return l1

    def __repr__(self):
        return "Layer 1: {0}\nLayer 2: {1}".format(self.layer_1,self.layer_2)
        
    def update(self,layer_1,layer_2,flag):
        # we do not check if 2 layer are the same
        if flag == True:
            if len(layer_2) > 0:
                layer = self.__ewd_evolution(self.layer_2,layer_2)
                t = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
                t.decode(layer,self.network.distance,self.network.trans_range)
                if t.is_fisible(self.network.distance,self.network.trans_range):
                    self.layer_2 = layer
        else:
            if len(layer_1) > 0:
                self.__poo(self.layer_1,layer_1)
                self.layer_2 = self.__tree_evolution(self.layer_1, self.layer_2, layer_2)
    
    # ewd evolution in case flag == SAME
    def __ewd_evolution(self, p1_2, p2_2):
        # adjacent node crossover
        layer = self.__anx(p1_2, p2_2)
        # reciprocal exchange mutation
        self.__rem(layer)
        return layer

    def __adj_map(self,tree):
        m = defaultdict(lambda: [])
        for node in tree.find_nodes():
            p = tree.find_parent(node)
            if p is not None: m[node].append(p)
            m[node].extend(tree.tree[node])
        
        return m

    def __find_node(self,node,candidates):
        return min(candidates, key=lambda next_node: self.network.distance(next_node,node))

    def __anx(self, p1_2, p2_2):
        # construct tree from 2 layers
        p1_tree,p2_tree = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes),Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
        p1_tree.decode(p1_2,self.network.distance,self.network.trans_range)
        p2_tree.decode(p2_2,self.network.distance,self.network.trans_range)

        # construct node adjacent map from 2 tree
        p1_map = self.__adj_map(p1_tree)
        p2_map = self.__adj_map(p2_tree)
        
        layer = [None] * len(p1_2)
        
        # randomly select initial node
        node = p1_tree.root
        layer[0] = node

        for i in range(1,len(layer)):
            s = [x for x in p1_map[node] if x in set(p2_map[node])]
            v = list(set().union(p1_map[node],p2_map[node]))

            if len(s) > 0:
                next_node = self.__find_node(node,s)

                p1_map[node].remove(next_node)
                p1_map[next_node].remove(node)

                p2_map[node].remove(next_node)
                p2_map[next_node].remove(node)

                node = next_node
            else:
                if len(v) > 0:
                    next_node = self.__find_node(node,v)

                    if next_node in p1_map[node]:
                        p1_map[node].remove(next_node)
                        p1_map[next_node].remove(node)
                    else:
                        p2_map[node].remove(next_node)
                        p2_map[next_node].remove(node)

                    node = next_node
                else:
                    node = random.choice(p1_tree.find_nodes())
            
            layer[i] = node
       
        for node in set(p1_2):
            if node not in set(layer):
                counter = Counter(layer)
                r = random.choice([x for x in counter if counter[x] > 1])
                layer[layer.index(r)] = node
        
        return layer

    def __rem(self,layer):
        i1,i2 = random.sample(range(len(layer)),2)
        layer[i1],layer[i2] = layer[i2],layer[i1]

    # tree evolution in case flag == DIFF
    def __poo(self, p1_1, p2_1):
        i1,i2 = sorted(random.sample(range(len(p1_1)),2))
        p1_1[:] = p1_1[:i1] + p2_1[i1:i2] + p1_1[i2:]

    def __tree_evolution(self, p1_1, p1_2, p2_2):
        p1_tree,p2_tree = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes),Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)

        p1_tree.decode(p1_2,self.network.distance,self.network.trans_range)
        p2_tree.decode(p2_2,self.network.distance,self.network.trans_range)
        
        intersections = [(x,y) for (x,y) in p1_tree.find_fringe() if (x,y) in set(p2_tree.find_fringe())]
        unions = list(set().union(p1_tree.find_fringe(),p2_tree.find_fringe()))
        residuals = [edge for edge in unions if edge not in intersections]

        s_nodes = [node for i,node in enumerate(self.network.relays) if p1_1[i] == 1]
        edges = [[src,dst] for [src,dst] in residuals if src in s_nodes and dst in s_nodes]

        candidates = [edge for edge in residuals if edge not in edges]
        candidates.sort(key=lambda x:self.network.distance(x[0],x[1]))

        t = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
        t.kruskal(intersections + edges + candidates,self.network.N)
        
        layer_len = 2*(self.network.N - 1)
        layer = t.encode(layer_len)

        return layer

class Swarm:
    def __init__(self,network,swarm_size=20,max_iter=200,max_consecutive=20,c0=0.25,c1=0.25,min_err=0.01):
        self.network = network
        # init some constant variables
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.max_consecutive = max_consecutive
        self.c0 = c0
        self.c1 = c1
        self.min_err = min_err

        self.swarm = []

        # list of tuple
        # (energy_consumption,self.network_lifetime,convergence_time,communication_interference)
        # for each particle
        self.target = []
        self.zmax = [1e-9] * 4
        self.zmin = [1e9] * 4

        for _ in range(swarm_size):
            p = Particle(self.network)
            self.swarm.append(p)
            self.update_target(self.calculate_target(p))
        
        # print("Zmax = {0}\nZmin = {1}".format(self.zmax,self.zmin))

        self.fitness = [1e9] * swarm_size
        self.g_layer_1 = []
        self.g_layer_2 = []
        self.g_err = sys.maxsize

    def __repr__(self):
        return "Layer 1: {0}\nLayer 2: {1}\nFitness value: {2}".format(self.g_layer_1,self.g_layer_2,self.g_err)

    def calculate_target(self,particle):
        t = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
        t.decode(particle.layer_2,self.network.distance,self.network.trans_range)
        return (
            self.network.energy_consumption(t),
            self.network.network_lifetime(t),
            self.network.convergence_time(t),
            self.network.communication_interference(t)
        )
    
    def update_target(self,target):
        self.zmax[:] = [max(x,y) for (x,y) in zip(self.zmax,target)]
        self.zmin[:] = [min(x,y) for (x,y) in zip(self.zmin,target)]

    # check if solution is dominated
    def is_dominated(self,solution):
        flag = False
        for another_solution in self.target:
            if another_solution is not solution:
                flag = reduce((lambda x,y:x&y),[x<y for (x,y) in zip(another_solution,solution)])
                if flag:
                    return flag

        return flag                

    def fitness_evaluation(self,target):
        weighted_sum = sum([(target[j] - self.zmin[j])/(self.zmax[j]-self.zmin[j] + 1e-6) for j in range(4)])
        pareto = 1 if self.is_dominated(target) else 0
        return weighted_sum + pareto
    
    def eval(self):
        i=0;k=0
        
        start = time.time()
        while i < self.max_iter and k < self.max_consecutive:
            for j in range(self.swarm_size):
                r = random.uniform(0,1)

                if self.c0 < r and r <= self.c0 + self.c1:
                    flag = (self.swarm[j].layer_1 == self.swarm[j].p_layer_1)
                    self.swarm[j].update(self.swarm[j].p_layer_1,self.swarm[j].p_layer_2,flag)
                else:
                    flag = (self.swarm[j].layer_1 == self.g_layer_1)
                    self.swarm[j].update(self.g_layer_1,self.g_layer_2,flag)
                
                target = self.calculate_target(self.swarm[j])
                self.update_target(target)
                f = self.fitness_evaluation(target)

                if f < self.fitness[j]:
                    self.fitness[j] = f
                    self.swarm[j].p_layer_1 = self.swarm[j].layer_1
                    self.swarm[j].p_layer_2 = self.swarm[j].layer_2
            
            # update global
            _g_err = min(self.fitness)
            g_err_tmp = float(self.g_err)
            
            if _g_err < self.g_err:
                idx = self.fitness.index(_g_err)
                self.g_err = float(self.fitness[idx])
                self.g_layer_1 = list(self.swarm[idx].p_layer_1)
                self.g_layer_2 = list(self.swarm[idx].p_layer_2)

            if abs(self.g_err - g_err_tmp) < self.min_err:
                k += 1
            else:
                k = 0
            
            i += 1
        
        end = time.time()

        return {'time':end-start,'gens':i,'value':self.g_err}
        
        #t = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
        #t.decode(self.g_layer_2,self.network.distance,self.network.trans_range)

        #return t

# for _ in range(100):
    # s = Swarm(network)
    # t = s.eval()
    # if not t.is_fisible(network.distance,network.trans_range):
        # print('Not OK')
