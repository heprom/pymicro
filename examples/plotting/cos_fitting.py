#!/usr/bin/env python
import numpy as np, os
from matplotlib import pyplot as plt
from pymicro.math.fitting import fit

if __name__ == '__main__':
  '''Basic curve fitting example.'''
  x = np.linspace(-3.0, 3.0, 31)
  print x
  # generate some noisy data
  np.random.seed(13)
  y = np.cos(4*x/np.pi - 0.5) + 0.05*np.random.randn(len(x))

  F = fit(y, x, fit_type='Cosine')
  plt.plot(x, y, 'bo')
  plt.plot(x, F(x), 'k-')
  plt.xlim(-3, 3)

  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name
  plt.savefig(image_name, format='png')

  from matplotlib import image
  image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
