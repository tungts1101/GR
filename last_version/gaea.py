from steinertree import SteinerTree
import random
from helper import Helper
from collections import defaultdict, Counter
from functools import reduce

class Particle:
    def __init__(self,network):
        self.pos = SteinerTree(network)
        self.pos.random_init()
        self.objective = None

class Swarm:
    def __init__(self,network,swarm_size=50,generations=20,epochs=50,Pc=0.8,Pm=0.05):
        self.network = network
        self.swarm_size = swarm_size
        self.generations = generations
        self.Pc = Pc
        self.Pm = Pm
        self.epochs = epochs

        self.swarm = []
        self.fv = 1e9
        self.g_pos = None

        self.zmax = [1e-9 for _ in range(4)]
        self.zmin = [1e9 for _ in range(4)]


    def init_swarm(self):
        for _ in range(self.swarm_size):
            self.swarm.append(Particle(self.network))

    def mutate(self,particle):
        p = Particle(self.network)
        p.pos = Helper.mutate(particle.pos,self.network)
        assert(p.pos.is_fisible())  # can comment to boost running time
        return p
    
    def crossover(self,p1,p2):
        p = Particle(self.network)
        p.pos = Helper.crossover(p1.pos,p2.pos,self.network)
        assert(p.pos.is_fisible())  # can comment to boost running time
        return p

    def eval_by_func(self,f):
        for _ in range(self.generations):
            P = list(self.swarm)
            # mutation
            Q1 = [self.mutate(P[i]) for i in range(len(P)) if random.uniform(0,1) < self.Pm]

            # crossover
            c = [i for i in range(len(P)) if random.uniform(0,1) < self.Pc]
            random.shuffle(c)
            c = c[:len(c) - len(c)%2]
            Q2 = [self.crossover(P[i],P[j]) for (i,j) in zip (c[:len(c)//2],c[len(c)//2:])]

            # selection
            swarm = P + Q1 + Q2
            
            self.swarm = sorted(swarm,key=lambda particle:f(particle.pos))[:self.swarm_size]
    
    def epoch_eval(self):
        for ff in [self.network.energy_consumption,
                   self.network.network_lifetime,
                   self.network.convergence_time,
                   self.network.communication_interference]:
            self.eval_by_func(ff)
    
    def is_dominated(self,p1):
        for p2 in self.swarm:
            if self.dominates(p2,p1):
                return True

        return False

    def dominates(self,p1,p2):
        return reduce((lambda x,y:x&y),[x<y for (x,y) in zip(p1.objective,p2.objective)])

    def calculate_fitness(self):        
        for p in self.swarm:
            p.objective = (
                self.network.energy_consumption(p.pos),
                self.network.network_lifetime(p.pos),
                self.network.convergence_time(p.pos),
                self.network.communication_interference(p.pos) 
            )

        for i in range(self.swarm_size):
            self.zmax[:] = [max(x,y) for (x,y) in zip(self.zmax,self.swarm[i].objective)]
            self.zmin[:] = [min(x,y) for (x,y) in zip(self.zmin,self.swarm[i].objective)]

        for i in range(self.swarm_size):
            fv = self.fitness_evaluation(self.swarm[i])
            if self.fv > fv:
                self.fv = float(fv)
                self.g_pos = self.swarm[i].pos
    
    def fitness_evaluation(self,particle):
        objective = particle.objective
        weighted_sum = sum([(objective[j] - self.zmin[j] + 1e-6)/(self.zmax[j] - self.zmin[j] + 1e-6) for j in range(4)])
        pareto = 1 if self.is_dominated(particle) else 0
        return weighted_sum + pareto

    def eval(self):
        self.init_swarm()
        for e in range(self.epochs):
            self.epoch_eval()
            self.calculate_fitness()
            assert(self.g_pos.is_fisible())

            print("Epoch = {}, fitness value = {}".format(e,self.fv))
