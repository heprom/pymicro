'''The fastcrystal module defines quick-to-use functions
   for calculations on crystals and pole figure plotting.
'''
import numpy as np
from pymicro.crystal.microstructure import Orientation, Grain, Microstructure
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
		elif(len(rawdata[0] == 3)):
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
