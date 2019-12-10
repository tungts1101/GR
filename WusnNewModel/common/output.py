import json
import copy

from logzero import logger

from common.input import WusnInput, WusnConstants, distance
from common.point import SensorNode, RelayNode


class WusnOutput:
    def __init__(self, inp: WusnInput, mapping=None):
        if mapping is None:
            mapping = {}

        self.inp = inp
        self.mapping = mapping

    def assign(self, relay, sensor):
        self.mapping[sensor] = relay

    def to_dict(self):
        inp_dict = self.inp.to_dict()
        mapping = []
        for sn, rn in self.mapping.items():
            mapping.append({
                'sensor': sn.to_dict(),
                'relay': rn.to_dict()
            })

        d = {
            'input': inp_dict,
            'mapping': mapping
        }

        return d

    def copy(self):
        mp = copy.deepcopy(self.mapping)
        return WusnOutput(self.inp, mp)

    @property
    def used_relays(self):
        ur = set()
        for rn in self.mapping.values():
            ur.add(rn)
        return list(ur)

    def loss(self, alpha=0.5):
        ur = self.used_relays
        l1 = alpha * len(ur) / self.inp.num_of_relays

        l2 = -float('inf')
        for sn, rn in self.mapping.items():
            ls = self.inp.sensor_loss[(sn, rn)]
            if ls > l2:
                l2 = ls

        for rn in ur:
            conns = list(filter(lambda x: x == rn, self.mapping.values()))
            ls = WusnConstants.k_bit * (len(conns) * (WusnConstants.e_rx + WusnConstants.e_da) +
                                        WusnConstants.e_mp * (distance(rn, self.inp.BS) ** 4))
            if ls > l2:
                l2 = ls

        l2 = l2 / self.inp.e_max * (1 - alpha)

        return l1 + l2
    
    def max_consumption(self):
        ur = self.used_relays
        l2 = -float('inf')
        for rn in ur:
            conns = list(filter(lambda x: x == rn, self.mapping.values()))
            ls = WusnConstants.k_bit * (len(conns) * (WusnConstants.e_rx + WusnConstants.e_da) +
                                        WusnConstants.e_mp * (distance(rn, self.inp.BS) ** 4))
            if ls > l2:
                l2 = ls
        return l2
    
    def total_tranmission_loss(self):
        total = 0
        ur = self.used_relays
        for sn, rn in self.mapping.items():
            ls = self.inp.sensor_loss[(sn, rn)]
            total+=ls
        for rn in ur:
            conns = list(filter(lambda x: x == rn, self.mapping.values()))
            ls = WusnConstants.k_bit * (len(conns) * (WusnConstants.e_rx + WusnConstants.e_da) +
                                        WusnConstants.e_mp * (distance(rn, self.inp.BS) ** 4))
            total+=ls
            
        return total

    @classmethod
    def from_dict(cls, d):
        inp = WusnInput.from_dict(d['inp'])
        mapping = {}
        for val in d['mapping']:
            mapping[SensorNode.from_dict(val['sensor'])] = RelayNode.from_dict(val['relay'])

        return cls(inp, mapping)

    def to_file(self, path):
        with open(path, 'wt') as f:
            fstr = json.dumps(self.to_dict(), indent=4)
            f.write(fstr)

    @classmethod
    def from_file(cls, path):
        with open(path, 'rt') as f:
            d = json.load(f)
            return cls.from_dict(d)