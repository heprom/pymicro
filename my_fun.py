import numpy as np
from pymicro.crystal.microstructure import *
from pymicro.crystal.texture import *
from pymicro.crystal.lattice import *
from pymicro.crystal.fastcrystal import *
import matplotlib.pyplot as plt




def Write_inp_crystals(grain_list,Eul_file,filename):
	grains = np.genfromtxt(Eul_file)
	
	file = open(filename,'w')
	
	for i in range(np.shape(grain_list)[0]):
		n = grain_list[i]
		file.write('**elset grain_%d *file AlLi.mat *integration runge_kutta 0.000001 0.000001 *rotation  %06f %06f %06f \n' % (n, grains[n-1,0],grains[n-1,1],grains[n-1,2])) #-1 because grains[0,:] is for grain number 1
		print i
		 
	file.close()


def Eul2Mat(euler):
    '''
    Compute the orientation matrix associated with the 3 Euler angles 
    (given in degrees).
    '''
    (rphi1, rPhi, rphi2) = np.radians(euler)
    c1 = np.cos(rphi1)
    s1 = np.sin(rphi1)
    c = np.cos(rPhi)
    s = np.sin(rPhi)
    c2 = np.cos(rphi2)
    s2 = np.sin(rphi2)

    # rotation matrix B
    b11 = c1*c2 - s1*s2*c
    b12 = s1*c2 + c1*s2*c
    b13 = s2*s
    b21 = -c1*s2 - s1*c2*c
    b22 = -s1*s2 + c1*c2*c
    b23 = c2*s
    b31 = s1*s
    b32 = -c1*s
    b33 = c
    B = np.array([[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]])
    return B
    
def sst_symmetry_cubic( z_rot):
    '''Perform cubic symmetry according to the unit SST triangle.
    '''
    if z_rot[0] < 0: z_rot[0] = -z_rot[0]
    if z_rot[1] < 0: z_rot[1] = -z_rot[1]
    if z_rot[2] < 0: z_rot[2] = -z_rot[2]

    if (z_rot[2] > z_rot[1]):
      z_rot[1], z_rot[2] = z_rot[2], z_rot[1]
    
    if (z_rot[1] > z_rot[0]):
      z_rot[0], z_rot[1] = z_rot[1], z_rot[0]
      
    if (z_rot[2] > z_rot[1]):
	  z_rot[1], z_rot[2] = z_rot[2], z_rot[1]
      
    return [z_rot[1], z_rot[2], z_rot[0]]

def plot_line_between_crystal_dir( c1, c2, steps=11, col='k',proj='stereo'):

    _path = np.zeros((steps,2), dtype=float)
    for j, i in enumerate(np.linspace(0, 1, steps)):
      ci = i*c1 + (1-i)*c2
      ci /= np.linalg.norm(ci)
      if proj == 'stereo':
        ci += [0,0,1]
        ci /= ci[2]
      _path[j,0] = ci[0]
      _path[j,1] = ci[1]
    plot(_path[:,0], _path[:,1], color=col)


def plot_ipf_density(file):
	eul = euler_from_file(file)
	eul=eul[0][:]
	coord =np.zeros((np.shape(eul)[0],2))
	axis = [0,0,1]
	micro = Microstructure(name = 'micro')
	pf = PoleFigure(axis='Y', proj='stereo', microstructure=micro)
	
	for i in range(np.shape(eul)[0]):
		print i
		R = Eul2Mat(eul[i][:])
		R=R.transpose()
		#c_dir = pf.sst_symmetry_cubic(R.dot(axis))
		c_dir = R.dot(axis)
		if c_dir[2] < 0: c_dir *= -1 # make unit vector have z>0
		if pf.proj == 'flat':
			cp = c_dir
		elif pf.proj == 'stereo':
			c = np.array(c_dir) + np.array(axis)
			c /= c[2] # SP'/SP = r/z with r=1
			cp = c
		coord[i,0]=cp[0]
		coord[i,1]=cp[1]

		
	#	print np.shape(coord)
		

	
	x=coord[::,0]
	y=coord[::,1]
	#print np.shape(x)
	plt.plot(x,y,marker='.')
	plt.show()
	# Estimate the 2D histogram
	nbins = 40
	H, xedges, yedges = np.histogram2d(x,y,bins=nbins)
	 
	# H needs to be rotated and flipped
	H = np.rot90(H)
	H = np.flipud(H)
	 
	# Mask zeros
	Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
	 
	# Plot 2D histogram using pcolor
	fig2 = plt.figure()
	plt.pcolormesh(xedges,yedges,Hmasked)
	plt.xlabel('x')
	plt.ylabel('y')
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Counts')
	plt.show()


class Nfun:

	
	
	@staticmethod       
	def PlotCrystalTrajectory(Zfile,B,full_ipf = True):
		'''Function to plot the trajectory of a crystal direction in the SST
		Currently work to plot the [0, 1, 0] direction of a crystal'''
		data = np.genfromtxt(Zfile)
		
		size = np.shape(data)
		
		R11 = data[:,1]
		R22 = data[:,2]
		R33 = data[:,3]
		R12 = data[:,4]
		R23 = data[:,5]
		R31 = data[:,6]
		R21 = data[:,7]
		R32 = data[:,8]
		R13 = data[:,9]
		
		orientation = Orientation(B)
		micro = Microstructure(name = 'micro')
		grain = Grain(1,orientation)
		micro.grains.append(grain)
		print 'B',B
		pf = PoleFigure(axis='Y', proj='stereo', microstructure=micro)
		pf.verbose = True
		if pf.axis == 'Z':
			axis = pf.z
		elif pf.axis == 'Y':
			axis = pf.y
		else:
			axis = pf.x
		axis_rot = pf.sst_symmetry_cubic(B.dot(axis)) # init load dir
		
		#pf.plot_pole_figures()
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, aspect='equal')
		pf.plot_pf(ax1, ann=True)
		ax2 = fig.add_subplot(122, aspect='equal')
		if full_ipf:
			pf.plot_ipf(ax = ax2, mk='.', col='k', ann=True)
		else:
			pf.plot_sst(ax = ax2, mk='.', col='k', ann=False)
		pole = HklDirection(1,0,0).direction()

		for i in range(size[0]):
			rot = np.array([[R11[i],R12[i],R13[i]],[R21[i],R22[i],R23[i]],[R31[i],R32[i],R33[i]]])
			print 'rot',rot
			new_B = rot.transpose().dot(B) # seems to work better at least in 2D
			print 'newB',new_B
			if full_ipf:
				axis_rot = new_B.dot(axis)
			else:
				axis_rot = pf.sst_symmetry_cubic(new_B.dot(axis))
			print 'axis rot', axis_rot
			pf.plot_crystal_dir(axis_rot, mk='.', col='b', ax=ax2, ann=False)
			d_rot = new_B.dot(pole)
			print 'i=',i, 'd_rot=',d_rot
			# direct plot
			pf.plot_pf_dir(d_rot, ax=ax1, mk='.', col='b', ann=False)
		
		if full_ipf:
			pf.plot_crystal_dir(B.dot(axis), mk='o', col='r', ax=ax2, ann=True) 
		else:
			pf.plot_crystal_dir(axis_rot, mk='o', col='r', ax=ax2, ann=False) 
		print '***'
		pf.plot_pf_dir(B.dot(pole), ax=ax1, mk='.', col='r', ann=False)
		plt.show()
		
		



