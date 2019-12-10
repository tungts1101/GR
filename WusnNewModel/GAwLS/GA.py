import time, os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from .tabu_search import *

GEN = 1000
CP = 0.8
MP = 0.05
NUM_OF_INDIVIDUALS = 100
TERMINATE = 50

def random_init_individual(num_relay, Y):
    indi = []
    xs = Y/num_relay
    count_relay = 0
    for i in range(0, num_relay):
        xx = random.random()
        if xx < xs:
            indi.append(1)
            count_relay += 1
        else:
            indi.append(0)
    if count_relay != Y:
        indi = formalize(indi, Y, count_relay)
    # if count_relay > Y:
    #     print(count_relay)
    return indi

def count_current_relay(individual):
    sum = 0
    for g in individual:
        if g == 1:
            sum += 1
    return sum

def formalize(individual, Y, count_relay):
    "Number of relay maybe less or greater than Y"
    indi = individual[:]
    num_relay = len(indi)
    if count_relay > Y:
        for i in range(0, count_relay - Y):
            id = random.randint(0, num_relay-1)
            while indi[id] == 0:
                id = random.randint(0, num_relay-1)
            indi[id] = 0
    else:
        for i in range(0, Y - count_relay):
            id = random.randint(0, num_relay-1)
            while indi[id] == 1:
                id = random.randint(0, num_relay-1)
            indi[id] = 1
    return indi

# mom and dad instead of parent1 and parent2 =)
def cross(mom, dad, Y):
    num_relay = len(mom)
    mid = random.randint(0, num_relay-1)
    # child1.append(mom[:mid])
    # child1.append(dad[mid:])
    # child2.append(dad[:mid])
    # child2.append(mom[mid:])
    child1 = mom[:mid] + dad[mid:]
    child2 = dad[:mid] + mom[mid:]
    child1 = formalize(child1, Y, count_current_relay(child1))
    child2 = formalize(child2, Y, count_current_relay(child2))
    return child1, child2


def mutate(original):
    fake = original[:]
    ll = len(fake)
    id1 = random.randint(0, ll-1)
    while fake[id1] == 0:
        id1 = random.randint(0, ll-1)
    id2 = random.randint(0, ll-1)
    while fake[id2] == 1:
        id2 = random.randint(0, ll-1)
    fake[id1] = 0
    fake[id2] = 1
    return fake

def GA(inp: WusnInput) -> int:
    E0_sensor = [15]*inp.num_of_sensors
    E0_relay = [15]*inp.num_of_relays
    # Khoi tao quan the
    individuals = []

    # Cac ca the da duoc tinh toan
    calculated = {}

    for i in range (0, NUM_OF_INDIVIDUALS):
        indi = random_init_individual(inp.num_of_relays, inp.Y)
        config, c = tabu_search(inp, indi, E0_sensor, E0_relay)
        # calculated[str(indi)] = [config, c]
        individuals.append([indi, config, c])

        # print(individuals[0])
    
    count_stable = 0
    max_c = individuals[0][2]
    prev_max = individuals[0][2]

    # Iterate through generations
    for it in range(0, GEN):
        start = time.time()
        none = 0
        not_none = 0
        # Crossover and mutation
        for id1 in range(0, NUM_OF_INDIVIDUALS):
            id2 = 0
            xx = random.random()
            if xx < CP:
                id2 = random.randint(0, NUM_OF_INDIVIDUALS-1)
                while id2 == id1:
                    id2 = random.randint(0, NUM_OF_INDIVIDUALS-1)
                son, daughter = cross(individuals[id1][0], individuals[id2][0], inp.Y)

                if str(son) in calculated:
                    config1 = calculated[str(son)][0]
                    cost1 = calculated[str(son)][1]
                else:
                    s = time.time()
                    config1, cost1 = tabu_search(inp, son, E0_sensor, E0_relay)
                    t = time.time()
                    
                if str(daughter) in calculated:
                    config2 = calculated[str(daughter)][0]
                    cost2 = calculated[str(daughter)][1]
                else:
                    s = time.time()
                    config2, cost2 = tabu_search(inp, daughter, E0_sensor, E0_relay)
                    t = time.time()

                if config1 == None:
                    none += 1
                else: 
                    not_none += 1
                if config2 == None:
                    none += 1
                else:
                    not_none += 1

                # config1, cost1 = tabu_search(inp, son)
                # config2, cost2 = tabu_search(inp, daughter)
                individuals.append([son, config1, cost1])
                individuals.append([daughter, config2, cost2])
                xx2 = random.random()
                if xx2 < MP:
                    grand_child1 = mutate(son)
                    grand_child2 = mutate(daughter)
                    config1, cost1 = tabu_search(inp, grand_child1, E0_sensor, E0_relay)
                    config2, cost2 = tabu_search(inp, grand_child2, E0_sensor, E0_relay)
                    individuals.append([grand_child1, config1, cost1])
                    individuals.append([grand_child2, config2, cost2])
        individuals = sorted(individuals, key=lambda x: x[2], reverse=True)
        individuals = individuals[:NUM_OF_INDIVIDUALS]
        if individuals[0][2] < max_c:
            max_c = individuals[0][2]
        if individuals[0][2] == prev_max:
            count_stable += 1
        else:
            count_stable = 0
        if count_stable == TERMINATE:
            print("TERMINATE")
            break
        prev_max = individuals[0][2]
        end = time.time()
        print("none: %d, not_none: %d" % (none, not_none))
        print("Gen: %d, time: %fs, min: %f %f" % (it, end - start, individuals[0][2], individuals[99][2]))
    print(max_c)
    return individuals[0]

# Bo sung dieu kien SN va RN co ket noi duoc voi nhau hay khong bang cach them ban kinh sn va rn