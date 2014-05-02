import numpy as np
from matplotlib import pyplot as plt

'''
  Compute and plot the gray level histogram of the provided data array.
  Works only with 8 bit data (ie np.uint8).
'''
def hist(data, nb_bins=256, show=True, save=False, prefix='data', density=False):
  print 'computing gray level histogram'
  hist, bin_edges = np.histogram(data, bins=nb_bins, range=(0,255), density=density)
  bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
  plt.figure(1, figsize=(6,4))
  if density:
    plt.bar(bin_centers, 100*hist, width=1, fill=True, color='g', edgecolor='g')
    plt.ylabel('Probability (%)')
  else:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.bar(bin_centers, hist, width=1, fill=True, color='g', edgecolor='g')
    plt.ylabel('Counts')
  plt.xlim(0,256)
  plt.xlabel('8 bit gray level value')
  if save: plt.savefig(prefix + '_hist.png', format='png')
  if show: plt.show()

