import numpy as np
from pymicro.crystal.microstructure import *
from pymicro.crystal.texture import *
from pymicro.crystal.lattice import *


#@staticmethod
def plot_mypf(pf, ax=None, mk='o', col='k', ann=False):
	'''Create the direct pole figure. '''
	pf.plot_pf_background(ax)
	ax.annotate('x', (1.01, 0.0), xycoords='data',
	  fontsize=16, horizontalalignment='left', verticalalignment='center')
	ax.annotate('z', (0.0, 1.01), xycoords='data',
	  fontsize=16, horizontalalignment='center', verticalalignment='bottom')
	for grain in pf.microstructure.grains:
	  B = grain.orientation_matrix()
	  Bt = B.transpose()
	  for c in pf.c001s:
		label = ''
		c_rot = Bt.dot(c)
		if pf.verbose: print 'plotting ',c,' in sample CS:',c_rot
		if pf.color_by_grain_id:
		  col = Microstructure.rand_cmap().colors[grain.id]
		  if pf.pflegend and pf.c001s.tolist().index(c.tolist()) == 0:
			# only add grain legend for the first crystal direction
			label = 'grain ' + str(grain.id)
		plot_crystal_dir_y(pf, c_rot, mk=mk, col=col, ax=ax, ann=ann, lab=label)
	ax.axis([-1.1,1.1,-1.1,1.1])
	if pf.pflegend and pf.color_by_grain_id:
	  ax.legend(bbox_to_anchor=(0.05, 1), loc=1, numpoints=1, \
		prop={'size':10})
	ax.axis('off')
	ax.set_title('direct %s projection' % pf.proj)

	

def plot_crystal_dir_y(pf, c_dir, mk='o', col='k', ax=None, ann=False, lab=''):
    '''Helper function to plot a crystal direction.'''
    if c_dir[1] < 0: c_dir *= -1 # make unit vector have y<0
    if pf.proj == 'flat':
      cp = c_dir
    elif pf.proj == 'stereo':
      c = c_dir + pf.y
      c /= c[1] # SP'/SP = r/z with r=1
      cp = c
      #cp = np.cross(c, pf.z)
    else:
      raise TypeError('Error, unsupported projection type', proj)
    ax.plot(cp[0], cp[2], linewidth=0, markerfacecolor=col, marker=mk, \
      markersize=pf.mksize, label=lab)
    if ann:
      ax.annotate(c_dir.view(), (cp[0], cp[2]-0.1), xycoords='data',
        fontsize=8, horizontalalignment='center', verticalalignment='center')


class Nfun:
	
	@staticmethod
	def ComputeSchmidFactor(B,n,l,loaddirection):
	  ''' 
	  Compute The Schmid factor for a crystal oriented by B and 
	  a slip system{n}[b]
	  '''
	
	  n = HklPlane(n[0],n[1],n[2])
	  l = HklDirection(l[0],l[1],l[2])
	  s = SlipSystem(n,l)
	  print s
	  o = Orientation(B)
	  #micro = Microstructure(name='microstructure')
	  g = Grain(1,o)
	  SF = g.schmid_factor(s, load_direction=loaddirection)
	
	  s = 'Schmid factor = '+ repr(SF)
	  print s
	  return SF
	  
	  
	  
	@staticmethod
	def ComputeSchmidFactorFCC(B,loaddirection):
		''' 
		Compute The Schmid factors of all slip systems of a fcc crystal oriented
		by B
		'''
		slip_systems = Lattice.get_slip_systems()
		o = Orientation(B)
		g = Grain(1,o)
		SF = np.zeros(len(slip_systems))
	
		for i in range(len(slip_systems)):
			s = slip_systems[i]
			SF[i] = g.schmid_factor(s, load_direction=loaddirection)
			c = 'Schmid Factor of system '+str(s)+' is '+ str(SF[i])
			print c
			
		return SF
	
	        
	@staticmethod       
	def PlotCrystalTrajectory(Zfile,B,full_ipf = True):
		'''Function to plot the trajectory of a crystal direction in the SST
		Currently work to plot the [0, 1, 0] direction of a crystal'''
		data = np.genfromtxt(Zfile)
		
		size = np.shape(data)
		
		R11 = data[:,0]
		R22 = data[:,1]
		R33 = data[:,2]
		R12 = data[:,3]
		R23 = data[:,4]
		R31 = data[:,5]
		R21 = data[:,6]
		R32 = data[:,7]
		R13 = data[:,8]
		
		orientation = Orientation(B)
		micro = Microstructure(name = 'micro')
		grain = Grain(1,orientation)
		micro.grains.append(grain)
		
		pf = PoleFigure( proj='stereo', microstructure=micro)
		pf.verbose = True
		#pf.plot_pole_figures()
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, aspect='equal')
		plot_mypf(pf, ax1, ann=True)
		ax2 = fig.add_subplot(122, aspect='equal')
		if full_ipf:
			pf.plot_ipf(ax = ax2, mk='.', col='k', ann=True)
		else:
			pf.plot_sst(ax = ax2, mk='.', col='k', ann=False)
		pf.vec = np.array([1.,0.,0.])
		pf.tensile_dir = np.array([0., 1., 0.])
		
		# now draw a line to picture the grain rotation
		B = grain.orientation_matrix()

		old_vec = pf.sst_symmetry_cubic(B.dot(pf.tensile_dir))

		for i in range(size[0]):
			rot = np.array([[R11[i],R12[i],R13[i]],[R21[i],R22[i],R23[i]],[R31[i],R32[i],R33[i]]])
			new_B = rot.transpose().dot(B) # seems to work better at least in 2D

			if full_ipf:
				vec_rot = new_B.dot(pf.tensile_dir)
			else:
				vec_rot = pf.sst_symmetry_cubic(new_B.dot(pf.tensile_dir))
			pf.plot_crystal_dir(vec_rot, mk='.', col='b', ax=ax2, ann=False)
			d_rot = new_B.transpose().dot(pf.vec)
			print i, d_rot
			plot_crystal_dir_y(pf, d_rot, mk='.', col='b', ax=ax1, ann=False)
		
		if full_ipf:
			pf.plot_crystal_dir(B.dot(pf.tensile_dir), mk='o', col='r', ax=ax2, ann=True) 

		else:
			pf.plot_crystal_dir(old_vec, mk='o', col='r', ax=ax2, ann=False) 

		plt.show()
