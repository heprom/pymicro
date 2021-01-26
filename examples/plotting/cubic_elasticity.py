from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt, cm
import os
import numpy as np

print('plotting cubic elasticity...')
fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
ax = fig.gca(projection='3d')

c11 = 192340.
c12 = 163140.
c44 = 41950.
s11 = (c11 + c12) / (c11 ** 2 + c11 * c12 - 2 * c12 ** 2)
s12 = -c12 / (c11 ** 2 + c11 * c12 - 2 * c12 ** 2)
s44 = 1. / c44
'''
# for cubic gold
s11 =  2.347e-05
s12 = -1.077e-05
s44 =  2.384e-05

# for ti beta
s11 =  2.87e-05
s12 = -1.29e-05
s44 =  1.82e-05
'''
print('elastic compliance s11 =%.6f, s12 =%.6f, s44 =%.6f' % (s11, s12, s44))

beta = 2. * s11 - 2. * s12 - s44
alpha = s11
# theta = np.linspace(0, np.pi*2, 46)
# phi = np.linspace(-np.pi*2, np.pi/2, 91)
theta = np.linspace(0, np.pi * 2, 181)
phi = np.linspace(-np.pi * 2, np.pi / 2, 361)
theta, phi = np.meshgrid(theta, phi)

n1 = np.cos(theta) * np.sin(phi)
n2 = np.sin(theta) * np.sin(phi)
n3 = np.cos(phi)

# rho = alpha - (beta*(np.cos(theta)**2*np.sin(theta)**2*np.sin(phi)**4 + np.cos(phi)**2*np.sin(phi)**2));
rho = alpha - beta * (n1 ** 2 * n2 ** 2 + n3 ** 2 * (n1 ** 2 + n2 ** 2))
rho = 1. / rho

x = rho * np.sin(phi) * np.cos(theta)
y = rho * np.sin(phi) * np.sin(theta)
z = rho * np.cos(phi)
ax.plot_surface(x, y, z, rstride=2, cstride=2, cmap=cm.jet, edgecolor='k',
                linewidth=1.0, antialiased=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
image_name = os.path.splitext(__file__)[0] + '.png'
plt.savefig(image_name, format='png')

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)

print('done')
