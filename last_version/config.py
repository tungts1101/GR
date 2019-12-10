import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'../WusnNewModel'))

from common.input import WusnConstants as cf
from common.input import WusnInput as inp

time_slot = 1.5
Erx = cf.e_rx
Efs = cf.e_fs
Eda = cf.e_da
Emp = cf.e_mp
k = cf.k_bit
E = 50

