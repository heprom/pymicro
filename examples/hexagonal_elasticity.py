from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt, cm
import numpy as np

print 'plotting hexagonal elasticity...'
fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
ax = fig.gca(projection='3d')

# elastic constants for pure alpha-Titanium
c11 = 162000.
c12 =  92000.
c13 =  69000.
c33 = 180000.
c44 =  46700.
c66 = 0.5*(c11 - c12)
s11 = (-c13**2+c11*c33)/((c11-c12)*(-2*c13**2+(c11+c12)*c33))
s12 = (c13**2-c12*c33)/((c11-c12)*(-2*c13**2+(c11+c12)*c33))
s13 = c13/(2*c13**2-(c11+c12)*c33)
s33 = (c11+c12)/(-2*c13**2+(c11+c12)*c33)
s44 = 1./c44
s66 = 2*(s11-s12)
print 'elsatic stiffness s11 =',s11
print 'elsatic stiffness s12 =',s12
print 'elsatic stiffness s13 =',s13
print 'elsatic stiffness s33 =',s33
print 'elsatic stiffness s44 =',s44
print 'elsatic stiffness s66 =',s66

#theta = np.linspace(0, np.pi*2, 46)
#phi = np.linspace(-np.pi*2, np.pi/2, 91)
theta = np.linspace(0, np.pi*2, 181)
phi = np.linspace(-np.pi*2, np.pi/2, 361)
theta, phi = np.meshgrid(theta, phi)

n1 = np.cos(theta)*np.sin(phi)
n2 = np.sin(theta)*np.sin(phi)
n3 = np.cos(phi)

rho = s11 - ((s11 - s33)*n3**2 + (2*s11 - 2*s13 - s44)*(n1**2 + n2**2))*n3**2
rho = 0.001/rho;

x = rho*np.sin(phi)*np.cos(theta);
y = rho*np.sin(phi)*np.sin(theta);
z = rho*np.cos(phi);

ax.plot_surface(x, y, z, rstride=2, cstride=2, cmap=cm.jet, \
  linewidth=0.5, antialiased=True)
ax.set_xlabel('X (GPa)')
ax.set_ylabel('Y (GPa)')
ax.set_zlabel('Z (GPa)')
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_zlim(-150, 150)
#plt.savefig('cubic_elasticity.eps', format='eps')
plt.savefig('hexagonal_elasticity.pdf', format='pdf')
#plt.show()
print 'done'
