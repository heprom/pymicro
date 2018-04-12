import numpy as np
from pymicro.crystal.lattice import HklPlane
from pymicro.crystal.microstructure import Orientation

X = np.array([1, 0, 0])
Y = np.array([0, 1, 0])
Z = np.array([0, 0, 1])
orientation = Orientation.from_euler(np.array([142.8, 32.0, 214.4]))
HklPlane.plot_slip_traces(orientation, hkl='111', n_int=np.array([np.sqrt(2) / 2., np.sqrt(2) / 2., 0.]), view_up=Z)
print('\n** YZ plane')
HklPlane.plot_YZ_slip_traces(orientation, hkl='110')
print('\n** XZ plane')
HklPlane.plot_XZ_slip_traces(orientation, hkl='110')
print('\n** XY plane')
HklPlane.plot_XY_slip_traces(orientation, hkl='110')
