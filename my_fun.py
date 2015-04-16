import numpy as np
from pymicro.crystal.microstructure import *
from pymicro.crystal.texture import *
from pymicro.crystal.lattice import *

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
