'''The fastcrystal module defines quick-to-use functions
   for calculations on crystals and pole figure plotting.
'''
import numpy as np
from pymicro.crystal.microstructure import Orientation, Grain, Microstructure
from pymicro.crystal.lattice import *
from pymicro.crystal.texture import PoleFigure
from random import shuffle

#Lorsqu'on donne [115, 10, 30] comme orientation d'Euler,
#orientation renvoie [295, 10, 30]. -> Verifier dans quel
#intervalle on prend les valeurs des angles d'Euler.
#A voir dans la fonction OrientationMatrix2Euler

def Euler2Rodrigues(euler):
	'''
	Compute the rodrigues vector from the 3 euler angles (in degrees).
	
	*Parameter*
	
	**euler**: [phi1, Phi, phi2] euler angles
	'''
	o = Orientation.from_euler(euler)
	print o.rod
	return o.rod

def Rodrigues2Euler(rod):
	'''
	Compute the 3 euler angles (in degrees) from the rodrigues vector.
	
	*Parameter*
	
	**rod**: rodrigues vector
	'''
	o = Orientation.from_rodrigues(rod)
	print o.euler
	return o.euler

def SchmidFactor_from_Rod(rod):
	o = Orientation.from_rodrigues(Rod)
	print o.schmid_factor



def euler_from_file(file):
	f = open(file, "r")
	f_lines = f.readlines()
	f.close()

	#In case the file contains many orientations: shuffle euler orientations
	#is a way to avoid that the last lines of the file are always 
	#the last plotted (which would result in a visual overrepresentation
	#of certain Euler orientations)
	shuffle(f_lines)

	rawdata = []
	orientations = []
	greylevels = []

	for line in f_lines:
		if (line[0] != "#"): rawdata.append(line.split())

	if (len(rawdata) == 0):
		print "!!!  There is no orientation to plot  !!!"
	else:
		if(len(rawdata[0]) >= 4):
			for line in rawdata:
				orientations.append([float(line[0]), float(line[1]), float(line[2])])
				greylevels.append(float(line[3]))
			to_return = [orientations, greylevels]
		elif(len(rawdata[0]) == 3):
			for line in rawdata:
				orientations.append([float(line[0]), float(line[1]), float(line[2])])
				greylevels.append(-1.)
			to_return = [orientations, greylevels]
		else:
			print "!!!  File is corrupted: not enough Euler angles  !!!"
	print to_return
	return to_return


def plot_pole_figures(euler):
	'''
	Plot direct and inverse pole figures of an orientation
	from euler angles.
	
	*Parameter*
	
	**euler**: [phi1, Phi, phi2] euler angles
	'''
	o = Orientation.from_euler(euler)
	g = Grain(1, o)
	m = Microstructure()
	m.name = '%s_%s_%s' %(int(euler[0]), int(euler[1]), int(euler[2]))
	m.grains = [g]
	pf = PoleFigure(m)
	pf.plot_pole_figures()
	
def plot_pole_figures_from_file(file):
	euler_grey = euler_from_file(file)
	eulers = euler_grey[0]
	print eulers
	
	m = Microstructure()
	m.name = 'from_file'
	m.grains = []
	
	for euler in eulers:
		o = Orientation.from_euler(euler)
		g = Grain(1, o)
		m.grains.append(g)
	
	pf = PoleFigure(m)
	pf.plot_pole_figures()
	
'''
def Calculate_Omega_dct(Rod,a = 0.405,hkl,E):

	Calculate diffracting onmegas angle for a grain in a given orientation. 
	Note: I have juste impemanted for {111} planes, {200} should be too!
	
	*Parameter*
	
	**Rodrigues vector**: (Rod) Rodrigues vector of the given grain
	
	**interplanar spacing**: (a) interplanar spacing for the given plane family, by default for the aluminium
	
	**Energy**: (E) Energy of the beam in keV

	
	m = Microstructure()
	m.name = 'DCT'
	m.grains = []
	o = Orientation.from_rodrigues(Rod)
	g = Grain(1, o)
	m.grains.append(g)
	
    l = Lattice.cubic(a)
    p = HklPlane(hkl[0], hkl[1],hkl[2] , lattice=l)
    
    Grain.dct_omega_angles(g,p,E)
	
'''	
	
def plot_pole_figures_from_file_Rod(file):
	Rods = np.genfromtxt(file)

	print np.shape(Rods)
	
	m = Microstructure()
	m.name = 'from_file'
	m.grains = []
	
	for Rod in Rods:
		o = Orientation.from_rodrigues(Rod)
		g = Grain(1, o)
		m.grains.append(g)
	
	pf = PoleFigure(m)
	pf.plot_pole_figures()


def calc_poles_id11(pl):
	'''
	Compute the two tilts to align a plane normal (pl) expressed in lab coordinates
	with Z=[0 0 1] in lab coordinates for the id11 diffractometer (ie: samrx ->samry ->diffrz ->diffry)
	
	return a numpy array like [samrx,samry]
	'''
	
	def Rx(alpha):
		Mx = np.array([[1 ,0 ,0],[ 0, np.cos(alpha) ,-np.sin(alpha)],[ 0 , np.sin(alpha) ,np.cos(alpha)]])
		return Mx
	
	def Ry(alpha):
		My = np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]])
		return My
		
	samry = np.arctan(-pl[0]/pl[2])
	
	pl_tilted = Ry(samry).dot(pl)
	
	print 'tilted (once) plane normal is:'
	print pl_tilted
	
	samrx = np.arctan(pl_tilted[1]/pl_tilted[2])
	
	pl_tilted = Rx(samrx).dot(pl_tilted)
	
	print 'tilted (twice) plane normal is:'
	print pl_tilted
	
	samrx_deg = samrx*180/np.pi
	samry_deg = samry*180/np.pi
	
	print 'samrx is:'
	print samrx_deg
	
	print 'samry is:'
	print samry_deg
	
	print 'associated Rotation matrix is:'
	print Rx(samrx).dot(Ry(samry))
	
	return np.array([samrx_deg,samry_deg])
	


def Sam2Lab(xyzSam,Sx,Sy,Sz):
	
	'''
	 Return in lab coordinates the xyzA vector initially expressed in Labcoordinates
	 ***input***
	 xyzA : vector expressed in  sample coordinates
	 Sx: x axis of the sample expressed in Lab coordinates
	 Sy: y axis of the sample expressed in Lab coordinates
	 Sz: z axis of the sample expressed in Lab coordinates
	'''
	 
	M_L2S = np.array([Sx,Sy,Sz])
	xyzLab = xyzSam.dot(M_L2S)
	
	return xyzLab


def Lab2sam(xyzLab,Sx,Sy,Sz):
	
	'''
	 Return in lab coordinates the xyzA vector initially expressed in Labcoordinates
	 ***input***
	 xyzA : vector expressed in  sample coordinates
	 Sx: x axis of the sample expressed in Lab coordinates
	 Sy: y axis of the sample expressed in Lab coordinates
	 Sz: z axis of the sample expressed in Lab coordinates
	'''
	 
	M_L2S = np.array([Sx,Sy,Sz])
	M_S2M = np.linalg.inv(M_L2S)
	xyzSam = xyzLab.dot(M_S2L)
	
	return xyzSam
	

def Sam2Sam(xyzA,Ax,Ay,Az,Bx,By,Bz):
	
	xyzLab = Sam2Lab(xyzsam,Ax,Ay,Az)
	
	xyzB = Lab2Sam(xyzLab,Bx,By,Bz)
	
	return xyzB
