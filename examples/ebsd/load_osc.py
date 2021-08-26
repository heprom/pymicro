import os
from pymicro.crystal.ebsd import OimScan, OimPhase
from pymicro.crystal.lattice import Lattice
from matplotlib import pyplot as plt, cm, image
from config import PYMICRO_EXAMPLES_DATA_DIR

file_path = os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'ebsd_ti_beta_crack.osc')
scan = OimScan.from_file(file_path)

# the crystalline phases are not read from the OSC file format -> add them manually
phase = OimPhase(1)
phase.name = 'Titanium Beta 21S'
phase.formula = 'Ti-15Mo-3Nb-3Al-0.2Si'
lattice = Lattice.cubic(0.3306)
phase.set_lattice(lattice)
phase.categories = [0, 0, 0, 0, 0]
scan.phase_list.append(phase)

# compute IPF maps
scan.compute_ipf_maps()

# plot a composite figure with IQ signal and IPF on top
plt.imshow(scan.iq, cmap=cm.gray)
plt.imshow(scan.ipf001, alpha=0.5)

image_name = os.path.splitext(__file__)[0] + '.png'
print('writing %s' % image_name)
plt.savefig(image_name, format='png')

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)

