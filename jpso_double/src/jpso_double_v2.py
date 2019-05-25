import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from steinertree import Tree
import random
from collections import defaultdict, Counter
from functools import reduce
import time

class Particle:
    def __init__(self,network):
        self.network = network
        self.solution = None

    def construct(self):
        edges = self.network.find_edges()
        random.shuffle(edges)
        random.shuffle(edges)
        t = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
        t.kruskal(edges,self.network.N)
        self.solution = t

    def __repr__(self):
        return "{}\n".format(self.solution.tree)

class Swarm:
    def __init__(self,network,swarm_size=20,max_iter=150,Pc=0.8,Pm=0.05,delta=0.01):
        self.network = network
        # init some constant variables
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.Pc = Pc    # probability of crossover
        self.Pm = Pm    # probability of mutate
        self.delta = delta

        self.swarm = []

        # list of tuple
        # (energy_consumption,network_lifetime,convergence_time,communication_interference)
        # for each particle
        self.target = []
        self.zmax = [1e-9] * 4
        self.zmin = [1e9] * 4
        
        self.fitness = []
        self.g_err = None

        for _ in range(swarm_size):
            p = Particle(self.network)
            p.construct()
            self.swarm.append(p)

        self.__update_global()

    def __repr__(self):
        # return "Best solution = {}".format(min(self.fitness))
        return ""
    
    def __update_global(self):
        self.target = []
        self.zmax = [1e-9] * 4
        self.zmin = [1e9] * 4
    
        for particle in self.swarm:
            target = self.calculate_target(particle)
            self.target.append(target)
            self.update_target(target)
        
        self.fitness = []
        for target in self.target:
            fv = self.fitness_evaluation(target)
            self.fitness.append(self.fitness_evaluation(target))
        
        idx = self.fitness.index(min(self.fitness))
        self.solution = self.swarm[idx].solution
        self.g_err = float(self.fitness[idx])

    def calculate_target(self,p):
        return (
            self.network.energy_consumption(p.solution),
            self.network.network_lifetime(p.solution),
            self.network.convergence_time(p.solution),
            self.network.communication_interference(p.solution)
        )
    
    def update_target(self,target):
        self.zmax[:] = [max(x,y) for (x,y) in zip(self.zmax,target)]
        self.zmin[:] = [min(x,y) for (x,y) in zip(self.zmin,target)]

    # check if solution is dominated
    def is_dominated(self,solution):
        flag = False
        for another_solution in self.target:
            if another_solution != solution:
                flag = reduce((lambda x,y:x&y),[x<y for (x,y) in zip(another_solution,solution)])
                if flag:
                    return flag

        return flag

    def fitness_evaluation(self,target):
        weighted_sum = sum([(target[j] - self.zmin[j] + 1e-6)/(self.zmax[j] - self.zmin[j] + 1e-6) for j in range(4)])
        pareto = 1 if self.is_dominated(target) else 0
        return weighted_sum + pareto
    
    def __select(self,q,r):
        if r < q[0]:
            return 0
        for i in range(1,len(q)):
            if q[i-1] < r and r <= q[i]:
                return i

    def __cross(self,p1,p2):
        p = Particle(self.network)

        t1 = p1.solution
        t2 = p2.solution
        t1_fringe = t1.find_fringe()
        t2_fringe = t2.find_fringe()

        intersections = [(x,y) for (x,y) in t1_fringe if (x,y) in t2_fringe]
        unions = list(set().union(t1_fringe,t2_fringe))
        
        unions.sort(key=lambda x:self.network.distance(x[0],x[1]))

        t = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
        t.kruskal(intersections + unions,self.network.N)

        p.solution = t
        return p

    def __mutate_helper(self,tree,fringe):
        def adj(tree,node):
            return [x if y is node else y for (x,y) in tree.find_fringe() if node in (x,y)]

        (begin,end) = fringe

        fringes = []
        fringes.append([begin])
        visited = []
        path = []

        while len(fringes) > 0:
            path = fringes.pop(0)
            node = path[-1]
            if node not in visited:
                visited.append(node)
                if node is end:
                    break

                for n in adj(tree,node):
                    if n not in visited:
                        new_path = list(path)
                        new_path.append(n)
                        fringes.append(new_path)

        redundant = [(x,y) for (x,y) in tree.find_fringe() if x in path and y in path]

        fringes = list(set(tree.find_fringe()) - set([random.choice(redundant)])) + [fringe]
        t = Tree(root=self.network.sink,compulsory_nodes=self.network.comp_nodes)
        t.kruskal(fringes,self.network.N)
        return t

    def __mutate(self,particle):
        p = Particle(particle.network)
        t = particle.solution
        i = 0
        x,y = random.sample(t.find_nodes(),2)
        while self.network.distance(x,y) > self.network.trans_range:
            x,y = random.sample(t.find_nodes(),2)
        
        p.solution = self.__mutate_helper(t,(x,y))
        return p

    def eval(self):
        i,k = 0,0
        
        start = time.time()

        while i < self.max_iter and k < 20:
            #pred_fv = sum(self.fitness)/len(self.fitness)
            pred_fv = float(self.g_err)

            P = list(self.swarm)
            
            # muration
            Q1 = [self.__mutate(P[i]) for i in range(len(P)) if random.uniform(0,1) < self.Pm]

            # crossover
            c = [i for i in range(len(P)) if random.uniform(0,1) < self.Pc]
            random.shuffle(c)
            c = c[:len(c) - len(c)%2]
            Q2 = [self.__cross(P[i],P[j]) for (i,j) in zip (c[:len(c)//2],c[len(c)//2:])]

            # selection
            swarm = P + Q1 + Q2

            #evals = []
            for particle in swarm:
                target = self.calculate_target(particle)
                self.update_target(target)
                #evals.append(self.fitness_evaluation(target))
            
            # F = sum([1/(x+self.delta) for x in evals])
            # p = [(1/(x+self.delta))/F for x in evals]
            # q = [sum(p[:x+1]) for x in range(len(swarm))]
            
            self.swarm = sorted(swarm,key=lambda particle:self.fitness_evaluation(self.calculate_target(particle)))[:20]

            #self.swarm = [swarm[self.__select(q,random.uniform(0,1))] for _ in range(len(swarm))][:20]
           
            self.__update_global()
            #cur_fv = sum(self.fitness)/len(self.fitness)
            cur_fv = float(self.g_err)

            if abs(cur_fv - pred_fv) < self.delta:
                k += 1
            else:
                k = 0

            i += 1

        end = time.time()
        
        result = {
            'time': end - start,
            'gen': i,
            'value': self.g_err
        }
        return result 
