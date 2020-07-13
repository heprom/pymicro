import os, numpy as np
from pymicro.crystal.texture import PoleFigure
from pymicro.crystal.microstructure import Microstructure, Grain, Orientation
from matplotlib import pyplot as plt

if __name__ == '__main__':
    '''
    A pole figure plotted using contours.
    '''
    eulers = Orientation.read_orientations('../data/pp100', data_type='euler')
    micro = Microstructure(name='test')
    for i in range(100):
        micro.grains.append(Grain(i + 1, eulers[i + 1]))

    pf = PoleFigure(microstructure=micro)
    pf.mksize = 40
    pf.proj = 'stereo'
    pf.set_hkl_poles('111')
    fig = plt.figure(1, figsize=(12, 5))
    ax1 = fig.add_subplot(121, aspect='equal')
    ax2 = fig.add_subplot(122, aspect='equal')
    pf.create_pf_contour(ax=ax1, ang_step=20)
    pf.plot_pf(ax=ax2)

    image_name = os.path.splitext(__file__)[0] + '.png'
    print('writting %s' % image_name)
    plt.savefig(image_name, format='png')

    from matplotlib import image

    image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
