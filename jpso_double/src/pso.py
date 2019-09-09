import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from steinertree import Tree
import random
from collections import defaultdict, Counter
from functools import reduce
import time
from multiprocessing import Process,Manager

def get_solution(network,layer):
    t = Tree(root=network.sink,compulsory_nodes=network.terminals)
    t.decode(layer,network.distance,network.trans_range)
    return t

class Particle:
    def __init__(self,network):
        self.network = network

        # current position & current error
        l2 = self.__gen_l2()
        l1 = self.__gen_l1_from_l2(l2)
        self.c_pos = [l1,l2]
        self.eval()
        
        # best position & best error
        self.b_pos = self.c_pos
        self.objective, self.c_err, self.b_err = None, None, None

    def __gen_l2(self):
        # create a fisible solution
        t = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)
        # find all eligible edges
        edges = self.network.find_edges()
        # shuffle edges to make our swarm more diversified
        random.shuffle(edges)
        t.kruskal(edges,self.network.N)
        if not t.is_fisible(self.network.distance,self.network.trans_range):
            print('Fail when create a tree in partice')
            exit(0)
        
        l2_len = 2*(self.network.N - 1)
        l2 = t.encode(l2_len)
        del t
        return l2
    
    def __gen_l1_from_l2(self,l2):
        l1 = []
        for i in range(self.network.N):
            if i not in self.network.terminals:
                if i in set(l2):
                    l1.append(1)
                else:
                    l1.append(0)
        
        return l1

    def fly(self,pos,flag):
        offspring = []

        if flag == True:
            o_1,b_2 = pos
            
            p_2 = self.c_pos[1]
            o_2 = self.__ewd_evol(p_2,b_2)
            
            offspring = [o_1,o_2]
        else:
            b_1,b_2 = pos
            p_1,p_2 = self.c_pos

            o_1 = self.__poo(p_1,b_1)
            o_2 = self.__tree_evol(o_1,p_2,b_2)

            offspring = [o_1,o_2]

        return offspring

    def __ewd_evol(self,p1,p2):
        def adj_map(tree):
            m = defaultdict(lambda: [])
            for node in tree.find_nodes():
                p = tree.find_parent(node)
                if p is not None: m[node].append(p)
                m[node].extend(tree.tree[node])
            
            return m

        def find_node(node,candidates):
            return min(candidates, key=lambda next_node: self.network.distance(next_node,node))

        def anx(p1,p2):
            # construct tree from 2 layers
            p1_tree,p2_tree = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals),Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)
            p1_tree.decode(p1,self.network.distance,self.network.trans_range)
            p2_tree.decode(p2,self.network.distance,self.network.trans_range)

            # construct node adjacent map from 2 tree
            p1_map = adj_map(p1_tree)
            p2_map = adj_map(p2_tree)
            
            layer = [None] * len(p1)
            
            # randomly select initial node
            node = p1_tree.root
            layer[0] = node

            for i in range(1,len(layer)):
                s = [x for x in p1_map[node] if x in set(p2_map[node])]
                v = list(set().union(p1_map[node],p2_map[node]))

                if len(s) > 0:
                    next_node = find_node(node,s)

                    p1_map[node].remove(next_node)
                    p1_map[next_node].remove(node)

                    p2_map[node].remove(next_node)
                    p2_map[next_node].remove(node)

                    node = next_node
                else:
                    if len(v) > 0:
                        next_node = find_node(node,v)

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
           
            for node in set(p1):
                if node not in set(layer):
                    counter = Counter(layer)
                    r = random.choice([x for x in counter if counter[x] > 1])
                    layer[layer.index(r)] = node
            
            return layer
        
        def rem(layer):
            i1,i2 = random.sample(range(len(layer)),2)
            layer[i1],layer[i2] = layer[i2],layer[i1]

        offspring = anx(p1,p2)
        rem(offspring)

        t = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)
        t.decode(offspring,self.network.distance,self.network.trans_range)
        if t.is_fisible(self.network.distance,self.network.trans_range):
            return offspring
        else:
            return p1
    
    def __poo(self,p1,p2):
        i1,i2 = sorted(random.sample(range(len(p1)),2))
        o1 = p1[:i1] + p2[i1:i2] + p1[i2:]
        return o1

    def __tree_evol(self, p1_1, p1_2, p2_2):
        p1_tree,p2_tree = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals),Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)

        p1_tree.decode(p1_2,self.network.distance,self.network.trans_range)
        p2_tree.decode(p2_2,self.network.distance,self.network.trans_range)
        
        intersections = [(x,y) for (x,y) in p1_tree.find_fringe() if (x,y) in set(p2_tree.find_fringe())]
        unions = list(set().union(p1_tree.find_fringe(),p2_tree.find_fringe()))
        residuals = [edge for edge in unions if edge not in intersections]

        s_nodes = [node for i,node in enumerate(self.network.relays) if p1_1[i] == 1]
        edges = [[src,dst] for [src,dst] in residuals if src in s_nodes and dst in s_nodes]

        candidates = [edge for edge in residuals if edge not in edges]
        candidates.sort(key=lambda x:self.network.distance(x[0],x[1]))

        t = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)
        t.kruskal(intersections + edges + candidates,self.network.N)
        
        layer_len = 2*(self.network.N - 1)
        layer = t.encode(layer_len)
        del t
        return layer

    def repair(self):
        pass

    # update current objective
    def eval(self):
        t = get_solution(self.network,self.c_pos[1])
        
        self.objective = [
            self.network.energy_consumption(t),
            self.network.network_lifetime(t),
            self.network.convergence_time(t),
            self.network.communication_interference(t)
        ]
    
    # udpate individual best
    # zmax,zmin stores maximum and minimum value
    # of each objective
    # flag is True if particle is dominated in Pareto
    # otherwise is False
    def update(self,zmin,zmax,flag):
        self.c_err = self.fv(zmax,zmin,flag)

        if self.b_err == None or self.c_err < self.b_err:
            self.b_pos = self.c_pos
            self.b_err = float(self.c_err)
    
    # fitness evaluation
    def fv(self,zmax,zmin,flag):
        # add 1e-6 in numerator and denominator to handle
        # zero case
        weighted_sum = sum([(self.objective[j] - zmin[j] + 1e-6)/(zmax[j]-zmin[j] + 1e-6) for j in range(len(self.objective))])
        pareto = 1 if flag else 0
        return weighted_sum + pareto

class Swarm:
    def __init__(self,network,swarm_size=50,generations=150,stall_gen=20,delta=0.01,C0=0,C1=0.5):
        self.network = network
        self.swarm_size = swarm_size
        self.generations = generations
        self.stall_gen = stall_gen
        self.delta = delta
        self.C0 = C0
        self.C1 = C1

        # global best position & global best error
        self.g_pos = []
        self.g_err = 1e9
        self.zmin = [1e9] * 4
        self.zmax = [1e-9] * 4
        self.cur_gen = 1
        self.cur_sg = 0
        self.running_time = 0
        
        self.swarm = []
        for _ in range(self.swarm_size):
            p = Particle(network)
            p.eval()
            self.update_objective(p.objective)
            self.swarm.append(p)

        for p in self.swarm:
            flag = self.is_dominated(p)
            p.update(self.zmax,self.zmin,flag)
        
        # update global best position & global best error
        self.update_global()

    def is_dominated(self,p1):
        for p2 in self.swarm:
            if self.dominates(p2,p1):
                return True

        return False

    def dominates(self,p1,p2):
        return reduce((lambda x,y:x&y),[x<y for (x,y) in zip(p1.objective,p2.objective)])

    def update_objective(self,objective):
        self.zmax[:] = [max(x,y) for (x,y) in zip(self.zmax,objective)]
        self.zmin[:] = [min(x,y) for (x,y) in zip(self.zmin,objective)]
    
    def can_stop(self):
        return self.cur_gen > self.generations or self.cur_sg > self.stall_gen or self.running_time > 15*60000

    def update_global(self):
        for particle in self.swarm:
            if particle.b_err < self.g_err:
                self.g_pos = particle.b_pos
                self.g_err = float(particle.b_err)
        
    def eval(self):
        start = time.time()

        while not self.can_stop():
            prv_g_err = float(self.g_err)

            # particle flies in here
            for particle in self.swarm:
                R = random.randint(0,1)

                if self.C0 < R and R <= self.C0 + self.C1:
                    # compare 2 layer 1 in current position
                    # and best position of particle j
                    p_layer_1 = particle.c_pos[0]
                    b_layer_1 = particle.b_pos[0]
                    flag = (p_layer_1 == b_layer_1)
                    particle.c_pos = particle.fly(particle.b_pos,flag)
                else:
                    # same routine as above
                    # except we use global best instead of local best
                    p_layer_1 = particle.c_pos[0]
                    g_layer_1 = self.g_pos[0]
                    flag = (p_layer_1 == g_layer_1)
                    particle.c_pos = particle.fly(self.g_pos,flag)

                # repair particle
                particle.repair()

                # udpate c_objective in particle
                particle.eval()

                # update Zmin,Zmax
                self.update_objective(particle.objective)

                # update individual best
                flag = self.is_dominated(particle)
                particle.update(self.zmax,self.zmin,flag)

            # update global best
            self.update_global()
            
            cur_g_err = float(self.g_err)
            self.cur_sg = self.cur_sg + 1 if abs(cur_g_err - prv_g_err) < self.delta else 0
            self.cur_gen += 1
            self.running_time = time.time() - start

        if not get_solution(self.network,self.g_pos[1]).is_fisible(self.network.distance,self.network.trans_range):
            print('Not OK')
            exit(0)

        return {"error": self.g_err, "running_time": self.running_time, "generations": self.cur_gen}
