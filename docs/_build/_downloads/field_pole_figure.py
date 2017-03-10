from pymicro.crystal.microstructure import *
from pymicro.crystal.texture import *

if __name__ == '__main__':
    '''This example demonstrate how a field can be used to color each
    symbol on the pole figure with the :py:meth:~`pymicro.crystal.texture.set_map_field`
    method.
    '''
    orientations = Orientation.read_euler_txt('../data/orientation_set.inp')
    micro = Microstructure(name='field')
    for i in range(600):
        micro.grains.append(Grain(i, orientations[i + 1]))

    # load strain from dat files
    strain_field = np.genfromtxt('../data/strain_avg_per_grain.dat')[19, ::2]

    # build custom pole figures
    pf = PoleFigure(microstructure=micro)
    pf.mksize = 8
    pf.set_map_field('strain', strain_field, field_min_level=0.015, field_max_level=0.025)
    fig = plt.figure()
    # direct PF
    ax1 = fig.add_subplot(111, aspect='equal')
    pf.plot_pf(ax=ax1)
    plt.title('111 pole figure, cubic elasticity')
    plt.savefig('%s_pole_figure.png' % micro.name, format='png')

    image_name = os.path.splitext(__file__)[0] + '.png'
    print 'writting %s' % image_name

    from matplotlib import image

    image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
