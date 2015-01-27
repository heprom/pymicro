import numpy as np
from pymicro.crystal.microstructure import *
from pymicro.crystal.texture import *
from pymicro.crystal.lattice import *

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
	
	
	
