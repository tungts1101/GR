import os, sys, random
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from common.input import *
max_iteration = 70
max_stable = 20
tabu_len = 10

# Radius of relays and sensors
R = 39


def generate_connect_matrix(inp: WusnInput, individual):
    "Check if there is a sensor can't connect to any relay and whether sensor can connect to relay or not at the same time"
    connect = {}
    for sn in inp.sensors:
        count = 0
        for i in range (0, len(individual)):
            if individual[i] == 1:
                if distance(sn, inp.relays[i]) <= 2*R:
                    connect[(sn, inp.relays[i])] = 1
                    count += 1
        if count == 0:
            return False
    return connect
# OK for newest version


def generate_random_config(inp: WusnInput, individual, connect):
    # generate list of available relays
    rn = create_avalable_relay_list(individual)
    relay_of_sensor = [0] * inp.num_of_sensors  
    Y = inp.Y
    # Assign relay to sensor
    for i in range(0, inp.num_of_sensors):
        xx = random.randint(0, Y-1)
        while (inp.sensors[i], inp.relays[rn[xx]]) not in connect:
            xx = random.randint(0, Y-1)
        relay_of_sensor[i] = rn[xx]
    return relay_of_sensor
# OK for newest version


def create_avalable_relay_list(individual):
    available = []
    for i in range (0, len(individual)):
        if individual[i] == 1:
            available.append(i)
    return available
# OK for newest version


def get_cost(inp: WusnInput, config, E0_sensor, E0_relay):
    min_remain = float("inf")
    rn_frequency = {}
    for i in range (0, len(config)):
        sn = inp.sensors[i]
        rn = inp.relays[config[i]]
        loss = inp.sensor_loss[(sn, rn)]
        remain = E0_sensor[i] - loss
        if remain < min_remain:
            min_remain = remain
        if config[i] not in rn_frequency:
            rn_frequency[config[i]] = 1
        else:
            rn_frequency[config[i]] += 1
    # print(frequency)

    for rn_id in rn_frequency.keys():
        if (E0_relay[rn_id] - inp.relay_loss[inp.relays[rn_id]] * rn_frequency[rn_id]) < min_remain:
            min_remain = E0_relay[rn_id] - inp.relay_loss[inp.relays[rn_id]] * rn_frequency[rn_id]
    return min_remain
# OK for newest version


def swap_relay(inp: WusnInput, connect, config, available_relay, basic_cost, E0_sensor, E0_relay):
    "Find max"
    res = config[:]
    min_cost = basic_cost
    num_of_sensors = len(config)
    id1 = 0
    id2 = 0
    for i in range(0, num_of_sensors-1):
        for j in range(i+1, num_of_sensors):
            sn1 = inp.sensors[i]
            sn2 = inp.sensors[j]
            rn1 = inp.relays[config[i]]
            rn2 = inp.relays[config[j]]
            if (sn1, rn2) in connect and (sn2, rn1) in connect:
                loss1 = inp.sensor_loss[(sn1, rn2)]
                loss2 = inp.sensor_loss[(sn2, rn1)]
                remain1 = E0_sensor[i] - loss1
                remain2 = E0_sensor[j] - loss2
                mm = min(min_cost, remain1, remain2)
                if mm < min_cost:
                    min_cost = mm
                    id1 = i
                    id2 = j
    res[id1], res[id2] = res[id2], res[id1]
    return res, min_cost
# almost OK for newest version


def change_relay(inp: WusnInput, connect, config, available_relay, basic_cost, E0_sensor, E0_relay):
    "Find max"
    res = config[:]
    min_cost = basic_cost
    num_of_sensors = inp.num_of_sensors
    id = 0
    new_rn = config[0]
    for i in range (0, num_of_sensors):
        for rn_id in available_relay:
            if config[i] != rn_id and (inp.sensors[i], inp.relays[rn_id]) in connect:
                loss = inp.sensor_loss[(inp.sensors[i], inp.relays[rn_id])]
                remain1 = E0_sensor[i] - loss
                fre_rn = 1
                for rln in config:
                    if rn_id == rln:
                        fre_rn += 1
            
                E_rn = inp.relay_loss[inp.relays[rn_id]]*fre_rn
                remain2 = E0_relay[rn_id] - E_rn
                mm = min(min_cost, remain1, remain2)
                if mm < min_cost:
                    res2 = config[:]
                    res2[i] = rn_id
                    # print(str(get_cost(inp, res2)) + " " + str(mm))
                    min_cost = mm
                    id = i
                    new_rn = rn_id
    res[id] = new_rn
    return res, min_cost
# almost OK for newest version

def tabu_search(inp: WusnInput, individual, E0_sensor, E0_relay):
    "Find max"
    
    connect = generate_connect_matrix(inp, individual)
    if connect == False:
        return None, -float("inf")
    centa = inp.base_station
    available_relay = create_avalable_relay_list(individual)
    tabu_list = []
    
    current_config = generate_random_config(inp, individual, connect)

    best_config = current_config[:]
    best_cost = get_cost(inp, current_config, E0_sensor, E0_relay)
    count_stable = 0
    for i in range (0, max_iteration):
        # print("OK")
        current_cost = get_cost(inp, current_config, E0_sensor, E0_relay)
        res1, cost1 = swap_relay(inp, connect, current_config, available_relay, current_cost, E0_sensor, E0_relay)
        res2, cost2 = change_relay(inp, connect, current_config, available_relay, current_cost, E0_sensor, E0_relay)
        if cost1 > cost2:
            current_config = res1
            current_cost = cost1
        else:
            current_config = res2
            current_cost = cost2
        if current_cost == best_cost or current_config in tabu_list:
            count_stable += 1
            if count_stable == max_stable:
                # print("Restarting tabu search ----------------")
                current_config = generate_random_config(inp, individual, connect)
                count_stable = 0
        elif current_cost > best_cost:
            best_cost = current_cost
            best_config = current_config
            tabu_list.append(current_config)
            if len(tabu_list) > tabu_len:
                tabu_list.pop(0)
        
        # print("Step: %d, cost: %f, nic: %d" % (i, current_cost, count_stable))
    return best_config, best_cost
# Not OK


# Muc tieu: nang luong con lai nho nhat la lon nhat => find the smallest cost of get_cost  
# Can chinh lai cac ham co lien quan tieu hao cua relay
# individual eg: [0, 0, 1, 1, 0, 0, 1, 0, 0]
# config for above individual: [2, 3, 2, 2, 6, 3, 6]
