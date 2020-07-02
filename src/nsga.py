import random
from functools import cmp_to_key
import time
from helper import Helper
import config
import copy


class Particle:
    def __init__(self, network):
        self.dom_set = []
        self.dom_count = 0
        self.rank = None
        self.dist = None
        self.pos = Helper.init_solution(network)
        self.objective = None

    def __repr__(self):
        return "{}\n".format(self.pos)


class Swarm:
    def __init__(self, network, swarm_size=50, generation=150, Pm=0.05, Pc=0.8):
        self.network = network
        self.swarm_size = swarm_size
        self.generation = generation
        self.Pm = Pm
        self.Pc = Pc

        # objective functions
        self.obj_func = (
            self.network.energy_consumption,
            self.network.network_lifetime,
            self.network.convergence_time,
            self.network.communication_interference
        )
        #####

        # global value
        self.cur_gen = 1
        self.zmax = [1e-9] * config.NUM_OBJECTIVE
        self.zmin = [1e9] * config.NUM_OBJECTIVE
        self.swarm = []

    def calculate_objective(self, p):
        return (
            self.network.energy_consumption(p.pos),
            self.network.network_lifetime(p.pos),
            self.network.convergence_time(p.pos),
            self.network.communication_interference(p.pos)
        )

    def update_objective(self, objective):
        self.zmax[:] = [max(x, y) for (x, y) in zip(self.zmax, objective)]
        self.zmin[:] = [min(x, y) for (x, y) in zip(self.zmin, objective)]

    def is_dominated(self, p):
        for _p in self.swarm:
            if Helper.dominate(_p, p):
                return True

        return False

    def fitness_evaluation(self, particle):
        objective = particle.objective
        weighted_sum = \
            sum([(objective[j] - self.zmin[j] + 1e-6) / max(self.zmax[j] - self.zmin[j], 1e-6)
                 for j in range(config.NUM_OBJECTIVE)])
        pareto = 1 if self.is_dominated(particle) else 0
        return weighted_sum + pareto

    def can_stop(self):
        return self.cur_gen >= self.generation

    def eval(self):
        def fast_nondominated_sort(pop):
            fronts = [[]]

            for p1 in pop:
                p1.dom_count = 0
                p1.dom_set = []

                for p2 in pop:
                    if Helper.dominate(p1, p2):
                        p1.dom_set.append(p2)
                    elif Helper.dominate(p2, p1):
                        p1.dom_count += 1

                if p1.dom_count == 0:
                    p1.rank = 0
                    fronts[0].append(p1)

            curr = 0
            while curr < len(fronts):
                next_front = []
                for p1 in fronts[curr]:
                    for p2 in p1.dom_set:
                        p2.dom_count -= 1
                        if p2.dom_count == 0:
                            p2.rank = curr + 1
                            next_front.append(p2)
                curr += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def calculate_crowding_distance(pop):
            # set distance to 0 for all particles
            for p in pop:
                p.dist = 0

            zmax = [1e-9] * config.NUM_OBJECTIVE
            zmin = [1e9] * config.NUM_OBJECTIVE
            for i in range(config.NUM_OBJECTIVE):
                for p in pop:
                    zmax = [max(x, y) for (x, y) in zip(zmax, p.objective)]
                    zmin = [min(x, y) for (x, y) in zip(zmin, p.objective)]

                pop[0].dist, pop[len(pop) - 1].dist = 1e15, 1e15

                for j in range(1, len(pop) - 1):
                    pop[j].dist += \
                        (pop[j + 1].objective[i] - pop[j - 1].objective[i]) / max((zmax[i]-zmin[i]), 1e-6)

        def crowd_comparison_operator(x, y):
            return x.dist <= y.dist if x.rank == y.rank else x.rank <= y.rank

        def select_parents(fronts, pop_size):
            for f in fronts:
                calculate_crowding_distance(f)

            offspring, last_front = [], 0
            for front in fronts:
                if len(offspring) + len(front) < pop_size:
                    offspring.extend(front)
                    last_front += 1
                else:
                    break

            # fill up swarm with the current front
            remaining = pop_size - len(offspring)
            if remaining > 0:
                fronts[last_front].sort(key=cmp_to_key(crowd_comparison_operator))
                offspring += fronts[last_front][:remaining]

            return offspring

        def crossover(p1, p2):
            return Helper.crossover(p1.pos, p2.pos, self.network)

        def mutate(particle):
            return Helper.mutate(particle.pos, self.network)

        def crossover_and_mutation(pop, f_crossover, f_mutate):
            Q = []

            c = [i for i in range(len(pop)) if random.uniform(0, 1) < self.Pc]
            random.shuffle(c)
            c = c[:len(c) - len(c) % 2]

            for (i, j) in zip(c[:len(c) // 2], c[len(c) // 2:]):
                p1, p2 = pop[i], pop[j]

                child = copy.copy(p1)

                child.pos = f_crossover(p1, p2)

                if random.random() <= self.Pm:
                    child.pos = f_mutate(child)

                Q.append(child)

            return Q

        def calculate_objectives(pop):
            for p in pop:
                p.objective = self.calculate_objective(p)
                self.update_objective(p.objective)

        def init_population():
            s = [Particle(self.network) for _ in range(self.swarm_size)]
            return s

        self.swarm = init_population()
        children = crossover_and_mutation(self.swarm, crossover, mutate)

        start = time.time()

        if config.TRACE:
            print("Start NSGA with swarm_size = {0}, max generation = {1}"
                  .format(self.swarm_size, self.generation))

        while not self.can_stop():
            if config.TRACE:
                print("Generation = {0}".format(self.cur_gen))

            union = self.swarm + children
            calculate_objectives(union)
            fronts = fast_nondominated_sort(union)

            self.swarm = select_parents(fronts, self.swarm_size)
            children = crossover_and_mutation(self.swarm, crossover, mutate)

            self.cur_gen += 1

        running_time = time.time() - start

        if config.TRACE:
            print("End NSGA at generation = {0} with running time = {1}"
                  .format(self.cur_gen, running_time))
            print("Pareto front contains value: {0}"
                  .format([self.fitness_evaluation(p) for p in fronts[0]]))
            print("--------------------------------------------------------------------------------")
            print("")

        return {"running_time": running_time, "front": fronts[0]}