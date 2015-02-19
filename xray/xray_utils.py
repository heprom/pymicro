import os, numpy as np
from matplotlib import pyplot as plt

def plot_xray_trans(mat='Al', t=1.0, rho=1.0, display=False):
  '''Plot the transmitted intensity of a X-ray beam through a given material.
  
  **Parameters:**
  
  *mat* The material (Chemical composition, eg. 'Al')
  
  *t* thickness of the material (1.0 by default)
  
  *rho* density of the material (1.0 by default)
  
  *display* display the plot or save an image of the plot (False by default)
  '''
  path = os.path.dirname(__file__)
  mu_rho = np.genfromtxt(os.path.join(path, mat + '.txt'), usecols = (0, 1), comments='#')
  rho = 2.8 # density in g/cm^3
  energy = mu_rho[:,0]
  # apply Beer-Lambert
  trans = 100*np.exp(-mu_rho[:,1]*rho*t/10)
  plt.plot(energy, trans, 'b-', linewidth=3, markersize=10, label='%s %.1f mm' % (mat, t))
  plt.xlim(1,100)
  plt.grid()
  plt.legend(loc='upper left')
  plt.xlabel('Photon Energy (keV)')
  plt.ylabel('Transmission I/I_0 (%)')
  if display:
    plt.show()
  else:
    plt.savefig('xray_trans_' + mat + '.png')
