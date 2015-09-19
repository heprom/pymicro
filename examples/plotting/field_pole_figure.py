from pymicro.crystal.microstructure import *
from pymicro.crystal.texture import *

def myhot():
  '''from scitools/easyviz/vtk_new_.html'''
  lut = []
  inc = 0.01175
  i = 0
  r = 0.0; g = 0.0; b = 0.0
  while r <= 1.:
    lut.append([i, r, g, b, 1])
    r += inc; i += 1
  r = 1.
  while g <= 1.:
    lut.append([i, r, g, b, 1])
    g += inc; i += 1
  g = 1.
  while b <= 1:
    if i == 256: break
    lut.append([i, r, g, b, 1])
    b += inc; i += 1
  return lut

if __name__ == '__main__':
  '''This example demonstrate how a field can be used to color each 
  symbol on the pole figure. This should be unified at ome point with 
  the color_by_grain_id option.
  '''
  lut = myhot()
  orientations = Orientation.read_euler_txt('../data/orientation_set.inp')
  micro = Microstructure(name='field')
  for i in range(600):
    micro.grains.append(Grain(i, orientations[i+1]))

  # load strain from dat files
  strain_field = np.genfromtxt('../data/strain_avg_per_grain.dat')[19, ::2]

  # build custom pole figures
  pf = PoleFigure(microstructure=micro)
  pf.mksize = 8
  pf.color_by_strain_level = True
  pf.lut = lut
  fig = plt.figure()
  # direct PF
  ax1 = fig.add_subplot(111, aspect='equal')
  pf.plot_pf_hot(ax = ax1, min_level = 0.015, max_level = 0.025, strain_levels = strain_field)
  plt.title('111 pole figure, cubic elasticity')
  plt.savefig('%s_pole_figure.png' % micro.name, format='png')
  
  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name

  from matplotlib import image
  image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
