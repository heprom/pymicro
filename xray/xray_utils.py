import os, numpy as np
from matplotlib import pyplot as plt

def lambda_keV_to_nm(lambda_keV):
  return 1.2398 / lambda_keV

def lambda_keV_to_angstrom(lambda_keV):
  return 12.398 / lambda_keV

def lambda_nm_to_keV(lambda_nm):
  return lambda_nm / 1.2398

def lambda_angstrom_to_keV(lambda_angstrom):
  return lambda_angstrom / 12.398

def plot_xray_trans(mat='Al', ts=[1.0], rho=1.0, unit='keV', energy_lim=(1, 100), display=False):
  '''Plot the transmitted intensity of a X-ray beam through a given material.
  
  **Parameters:**
  
  *mat* The material (Chemical composition, eg. 'Al')
  
  *ts* a list of thickness values of the material in mm (1.0 by default)
  
  *rho* density of the material in g/cm^3 (1.0 by default)
  
  *unit* unit for the energy column (keV by default)
  
  *energy_lim* energy bounds in the plot (1, 100 by default)
  
  *display* display the plot or save an image of the plot (False by default)
  '''
  path = os.path.dirname(__file__)
  print path
  mu_rho = np.genfromtxt(os.path.join(path, mat + '.txt'), usecols = (0, 1), comments='#')
  energy = mu_rho[:,0]
  if unit == 'MeV':
    energy *= 1000
  for t in ts:
    # apply Beer-Lambert
    trans = 100*np.exp(-mu_rho[:,1]*rho*t/10)
    plt.plot(energy, trans, '-', linewidth=3, markersize=10, label='%s %.1f mm' % (mat, t))
  plt.xlim(energy_lim)
  plt.grid()
  plt.legend(loc='upper left')
  plt.xlabel('Photon Energy (keV)')
  plt.ylabel('Transmission I/I_0 (%)')
  if display:
    plt.show()
  else:
    plt.savefig('xray_trans_' + mat + '.png')
