import os, numpy as np
from matplotlib import pyplot as plt, cm
from pymicro.xray.xray_utils import *

data_dir = '../../pymicro/xray/data'
data_name = 'Ni_atom_scattering'
data_path = os.path.join(data_dir, data_name + '.txt')

ni_fatom = np.genfromtxt(data_path)


# Test new function :
test = atom_scat_factor_function(mat='Ni', sintheta_lambda_max=16.0, display=True)

