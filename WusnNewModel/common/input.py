import json
import math
import os
import pickle

from collections import defaultdict

from common.point import *


class WusnConstants:
    # Unit: J
    e_tx = 50 * 1e-9
    e_rx = 50 * 1e-9
    e_fs = 10 * 1e-12
    e_da = 5 * 1e-12
    e_mp = 0.0013 * 1e-12

    # Num of bits
    k_bit = 4000


class WusnInput:
    def __init__(self, _W=500, _H=500, _depth=1., _height=10., _num_of_relays=10, _num_of_sensors=50,
                 _radius=20., _relays=None, _sensors=None, _BS=None, static_relay_loss=None,
                 dynamic_relay_loss=None, sensor_loss=None, max_rn_conn=None):
        self.W = _W
        self.H = _H
        self.depth = _depth
        self.height = _height
        self.relays = _relays
        self.sensors = _sensors
        self.num_of_relays = _num_of_relays
        self.num_of_sensors = _num_of_sensors
        self.radius = _radius
        self.BS = _BS
        # self.get_loss()
        self.static_relay_loss = static_relay_loss
        self.dynamic_relay_loss = dynamic_relay_loss
        self.sensor_loss = sensor_loss
        self.max_rn_conn = max_rn_conn

        # if None in (static_relay_loss, dynamic_relay_loss, sensor_loss):
            # self.calculate_loss()
        # if max_rn_conn == None:
            # self.calculate_max_rn_conn()

    @classmethod
    def from_file(cls, path):
        f = open(path)
        d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d):
        W = d['W']
        H = d['H']
        depth = d['depth']
        height = d['height']
        num_of_relays = d['num_of_relays']
        num_of_sensors = d['num_of_sensors']
        radius = d['radius']
        relays = []
        sensors = []
        BS = Point.from_dict(d['center'])
        for i in range(num_of_sensors):
            sensors.append(SensorNode.from_dict(d['sensors'][i]))
        for i in range(num_of_relays):
            relays.append(RelayNode.from_dict(d['relays'][i]))

        return cls(W, H, depth, height, num_of_relays, num_of_sensors, radius, relays, sensors, BS)

    def freeze(self):
        self.sensors = tuple(self.sensors)
        self.relays = tuple(self.relays)

    def to_dict(self):
        return {
            'W': self.W, 'H': self.H,
            'depth': self.depth, 'height': self.height,
            'num_of_relays': self.num_of_relays,
            'num_of_sensors': self.num_of_sensors,
            'relays': list(map(lambda x: x.to_dict(), self.relays)),
            'sensors': list(map(lambda x: x.to_dict(), self.sensors)),
            'center': self.BS.to_dict(),
            'radius': self.radius
        }

    def to_file(self, file_path):
        d = self.to_dict()
        with open(file_path, "wt") as f:
            fstr = json.dumps(d, indent=4)
            f.write(fstr)

    def create_cache(self):
        loss_file_name = str(hash(self)) + ".loss"
        list_loss_file = os.listdir("cache")
        if loss_file_name in list_loss_file:
            print("Cache exist")
        else:
            print("Creating cache")
            f = open("cache/" + loss_file_name, "wb")
            self.calculate_loss()
            data = [self.relay_loss, self.sensor_loss]
            pickle.dump(data, f)
            f.close()

    def calculate_max_rn_conn(self):
        max_rn_conn = {}
        R = self.radius
        BS = self.BS

        for rn in self.relays:
            max_rn_conn[rn] = 0
            for sn in self.sensors:
                if distance(sn, rn) <= 2*R:
                    max_rn_conn[rn] += 1 
        self.max_rn_conn = max_rn_conn

    @property
    def e_max(self):
        vals = []
        vals.extend(self.sensor_loss.values())
        max_rloss = []
        for rn in self.relays:
            max_rloss.append(WusnConstants.k_bit * (self.num_of_sensors * (WusnConstants.e_rx + WusnConstants.e_da) +
                                                    WusnConstants.e_mp * (distance(rn, self.BS) ** 4)))
        vals.extend(max_rloss)
        return max(vals)

    def calculate_loss(self):
        sensor_loss = defaultdict(lambda: float('inf'))
        static_relay_loss = {}
        dynamic_relay_loss = {}
        R = self.radius
        BS = self.BS
        for sn in self.sensors:
            for rn in self.relays:
                if distance(sn, rn) <= 2 * R:
                    sensor_loss[(sn, rn)] = WusnConstants.k_bit * (
                            WusnConstants.e_tx + WusnConstants.e_fs * math.pow(distance(sn, rn), 2))

        for rn in self.relays:
            dynamic_relay_loss[rn] = WusnConstants.k_bit * (WusnConstants.e_rx + WusnConstants.e_da)
            static_relay_loss[rn] = WusnConstants.k_bit * WusnConstants.e_mp * math.pow(distance(rn, BS), 4)

        self.static_relay_loss = static_relay_loss
        self.dynamic_relay_loss = dynamic_relay_loss
        self.sensor_loss = sensor_loss

    def __hash__(self):
        return hash((self.W, self.H, self.depth, self.height, self.num_of_relays, self.num_of_sensors, self.radius,
                     tuple(self.relays), tuple(self.sensors)))

    def __eq__(self, other):
        return hash(self) == hash(other)


if __name__ == "__main__":
    inp = WusnInput.from_file('./data/small_data/dem1_0.in')
    print(inp.relays[0])
