import numpy as np
from matplotlib import pyplot as plt

'''
  Compute and plot the gray level histogram of the provided data array.
'''
def hist(data, nb_bins=256, show=True, save=False):
  print 'computing gray level histogram'
  hist, bin_edges = np.histogram(data, bins=nb_bins, range=(0,255))
  bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
  plt.figure(1, figsize=(6,4))
  plt.bar(bin_edges[0:-1], hist, width=256./nb_bins, fill=True, color='g', edgecolor='g')
  if save: plt.savefig(prefix + '_hist.png', format='png')
  if show: plt.show()

