import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from steinertree import Tree
import random
from collections import defaultdict, Counter
from functools import reduce
import time
from multiprocessing import Process,Manager

class Particle:
    def __init__(self,network):
        edges = network.find_edges()
        random.shuffle(edges)
        t = Tree(root=network.sink,compulsory_nodes=network.terminals)
        t.kruskal(edges,network.N)
        self.pos = t
        self.objective = None

    def __repr__(self):
        return "{}\n".format(self.pos.tree)

class Swarm:
    def __init__(self,network,swarm_size=50,generations=150,stall_gen=20,Pc=0.8,Pm=0.05,delta=0.01):
        self.network = network
        self.swarm_size = swarm_size
        self.generations = generations
        self.Pc = Pc    # probability of crossover
        self.Pm = Pm    # probability of mutate
        self.delta = delta
        self.stall_gen = stall_gen


        self.zmax = [1e-9] * 4
        self.zmin = [1e9] * 4
        self.g_err = 1e9
        self.g_pos = None
        self.cur_gen = 1
        self.cur_sg = 0
        self.running_time = 0
        
        self.swarm = []
        for _ in range(self.swarm_size):
            self.swarm.append(Particle(self.network))

    def calculate_objectives(self,pop):
        for p in pop:
            p.objective = self.calculate_objective(p)
            self.update_objective(p.objective)

    def update_global(self):
        for particle in self.swarm:
            fv = self.fitness_evaluation(particle)

            if fv < self.g_err:
                self.g_pos = particle.pos
                self.g_err = float(fv)
    
    def calculate_objective(self,p):
        return (
            self.network.energy_consumption(p.pos),
            self.network.network_lifetime(p.pos),
            self.network.convergence_time(p.pos),
            self.network.communication_interference(p.pos)
        )
    
    def update_objective(self,objective):
        self.zmax[:] = [max(x,y) for (x,y) in zip(self.zmax,objective)]
        self.zmin[:] = [min(x,y) for (x,y) in zip(self.zmin,objective)]

    def is_dominated(self,p1):
        for p2 in self.swarm:
            if self.dominates(p2,p1):
                return True

        return False

    def dominates(self,p1,p2):
        return reduce((lambda x,y:x&y),[x<y for (x,y) in zip(p1.objective,p2.objective)])

    def fitness_evaluation(self,particle):
        objective = particle.objective
        weighted_sum = sum([(objective[j] - self.zmin[j] + 1e-6)/(self.zmax[j] - self.zmin[j] + 1e-6) for j in range(4)])
        pareto = 1 if self.is_dominated(particle) else 0
        return weighted_sum + pareto
    
    def select(self,q,r):
        if r < q[0]:
            return 0
        for i in range(1,len(q)):
            if q[i-1] < r and r <= q[i]:
                return i

    def cross(self,p1,p2):
        p = Particle(self.network)

        t1 = p1.pos
        t2 = p2.pos
        t1_fringe = t1.find_fringe()
        t2_fringe = t2.find_fringe()

        intersections = [(x,y) for (x,y) in t1_fringe if (x,y) in t2_fringe]
        unions = list(set().union(t1_fringe,t2_fringe))
        
        unions.sort(key=lambda x:self.network.distance(x[0],x[1]))

        t = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)
        t.kruskal(intersections + unions,self.network.N)

        p.pos = t
        return p
    
    def mutate(self,particle):
        p = Particle(self.network)

        tree = particle.pos

        begin,end = random.sample(tree.find_nodes(),2)
        while self.network.distance(begin,end) > self.network.trans_range:
            begin,end = random.sample(tree.find_nodes(),2)

        def adj(tree,node):
            return [x if y is node else y for (x,y) in tree.find_fringe() if node in (x,y)]

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

        fringes = list(set(tree.find_fringe()) - set([random.choice(redundant)])) + [(begin,end)]
        t = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)
        t.kruskal(fringes,self.network.N)
        p.pos = t
        return p
    
    def can_stop(self):
        return self.cur_gen > self.generations or self.cur_sg > self.stall_gen or self.running_time > 15*60000

    def eval(self):
        start = time.time()

        while not self.can_stop():
            prv_g_err = float(self.g_err)

            P = list(self.swarm)
            
            # mutation
            Q1 = [self.mutate(P[i]) for i in range(len(P)) if random.uniform(0,1) < self.Pm]

            # crossover
            c = [i for i in range(len(P)) if random.uniform(0,1) < self.Pc]
            random.shuffle(c)
            c = c[:len(c) - len(c)%2]
            Q2 = [self.cross(P[i],P[j]) for (i,j) in zip (c[:len(c)//2],c[len(c)//2:])]

            # selection
            swarm = P + Q1 + Q2
            self.calculate_objectives(swarm)
            
            self.swarm = sorted(swarm,key=lambda particle:self.fitness_evaluation(particle))[:self.swarm_size]
            self.update_global()
            
            cur_fv = float(self.g_err)
            cur_g_err = float(self.g_err)
            self.cur_sg = self.cur_sg + 1 if abs(cur_g_err - prv_g_err) < self.delta else 0
            self.cur_gen += 1
            self.running_time = time.time() - start
 
        if not self.g_pos.is_fisible(self.network.distance,self.network.trans_range):
            print('Not OK')
            exit(0)

        return {"error": self.g_err, "running_time": self.running_time, "generations": self.cur_gen} 
