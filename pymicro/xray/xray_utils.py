import os, numpy as np
from matplotlib import pyplot as plt
from skimage.transform import radon
from math import *

densities = {'Li': 0.533,  # Z = 3
             'Be': 1.8450,  # Z = 4
             'C': 2.26,  # Z = 6, strongly depends on crystal structure
             'Mg': 1.738,  # Z = 12
             'Al': 2.6941,  # Z = 13
             'Ti': 4.530,  # Z = 22
             'V': 6.100,  # Z = 23
             'Cr': 7.180,  # Z = 24
             'Mn': 7.430,  # Z = 25
             'Fe': 7.874,  # Z = 26
             'Co': 8.900,  # Z = 27
             'Ni': 8.902,  # Z = 28
             'Cu': 8.960,  # Z = 29
             'Zn': 7.112,  # Z = 30
             'Ga': 7.877,  # Z = 31
             'Ge': 5.370,  # Z = 32
             'Nb': 8.550,  # Z = 41
             'Pb': 11.330,  # Z = 82
             'WC': 15.63, # Z(W) = 74
             }

def atom_scat_factor_function(mat='Al', sintheta_lambda_max=12, display=True):
    '''Compute and display the fit function of the atomic scattering factor.

    :param string mat: A string representing the material (e.g. 'Al')
    :param float sintheta_lambda_max: maximal value of sin theta / lambda
    :param bool display: display an image of the plot
    '''

    data_dir = '../../pymicro/xray/data'
    param = np.genfromtxt(os.path.join(data_dir, mat + '_fit_fatom'))
    print('Fit coefficient for', param[:,1])
    fit = []
    sintheta_lambda = np.linspace(0.0, sintheta_lambda_max, 100)

    for i in sintheta_lambda:
        fit_fatom = param[0, 1] * exp(-param[1, 1] * i ** 2) + param[2, 1] * exp(-param[3, 1] * i ** 2) + param[4, 1] * exp(-param[5, 1] * i ** 2) + param[6, 1] * exp(-param[7, 1] * i ** 2) + param[8, 1]
        fit.append(fit_fatom)
    print(fit)

    plt.figure()
    plt.plot(sintheta_lambda, fit, 'o-', label=mat)
    plt.legend(loc='upper right')
    plt.xlabel(r'$\sin\theta / \lambda (nm^{-1})$')
    plt.ylabel('Atomic scattering factor')
    plt.title('Fit function of : %s' % mat)
    if display:
        plt.show()
    return fit


def lambda_keV_to_nm(lambda_keV):
    '''Change the unit of wavelength from keV to nm.

    :param float lambda_keV: the wavelength in keV unit.
    :returns: the wavelength in nm unit.
    '''
    return 1.2398 / lambda_keV


def lambda_keV_to_angstrom(lambda_keV):
    '''Change the unit of wavelength from keV to angstrom.

    :param float lambda_keV: the wavelength in keV unit.
    :returns: the wavelength in angstrom unit.
    '''
    return 12.398 / lambda_keV


def lambda_nm_to_keV(lambda_nm):
    '''Change the unit of wavelength from nm to keV.

    :param float lambda_nm: the wavelength in nm unit.
    :returns: the wavelength in keV unit.
    '''
    return 1.2398 / lambda_nm


def lambda_angstrom_to_keV(lambda_angstrom):
    '''Change the unit of wavelength from angstrom to keV.

    :param float lambda_angstrom: the wavelength in angstrom unit.
    :returns: the wavelength in keV unit.
    '''
    return 12.398 / lambda_angstrom


def plot_xray_trans(mat='Al', ts=[1.0], rho=None, energy_lim=[1, 100], legfmt='%.1f', display=True):
    '''Plot the transmitted intensity of a X-ray beam through a given material.

    This function compute the transmitted intensity from tabulated data of
    the mass attenuation coefficient \mu_\rho (between 1 and 100 keV) and
    applying Beer's Lambert law:

    .. math::

      I/I_0 = \exp(-\mu_\rho*\rho*t)

    The tabulated data is stored in ascii files in the data folder. It has been retrieved
    from NIST `http://physics.nist.gov/cgi-bin/ffast/ffast.pl`
    The density is also tabulated and can be left blanked unless a specific value is to be used.

    :param string mat: A string representing the material (e.g. 'Al')
    :param list ts: a list of thickness values of the material in mm ([1.0] by default)
    :param float rho: density of the material in g/cm^3 (None by default)
    :param list energy_lim: energy bounds in keV for the plot (1, 100 by default)
    :param string legfmt: string to format the legend plot
    :param bool display: display or save an image of the plot (False by default)
    '''
    path = os.path.dirname(__file__)
    print(path)
    mu_rho = np.genfromtxt(os.path.join(path, 'data', mat + '.txt'), usecols=(0, 1), comments='#')
    print('Data :', mu_rho)
    energy = mu_rho[:, 0]
    print('Energy :', energy)
    # look up density
    if rho is None:
        rho = densities[mat]
    legstr = '%%s %s mm' % legfmt
    plt.figure()
    for t in ts:
        # apply Beer-Lambert
        trans = 100 * np.exp(-mu_rho[:, 1] * rho * t / 10)
        plt.plot(energy, trans, '-', linewidth=3, markersize=10, label=legstr % (mat, t))
    # bound the energy to (1, 200)
    if energy_lim[0] < 1:
        energy_lim[0] = 1
    if energy_lim[1] > 200:
        energy_lim[1] = 200
    plt.xlim(energy_lim)
    #plt.ylim(0, 10)
    plt.grid()
    plt.legend(loc='upper left')
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Transmission I/I0 (%)')
    plt.title('Transmitted intensity of : %s' % mat)
    if display:
        plt.show()
    else:
        plt.savefig('xray_trans_' + mat + '.png')

def radiograph(data, omega):
    """Compute a single radiograph of a 3D object using the radon transform.

    :param np.array data: an array representing the 3D object in (XYZ) form.
    :param omega: the rotation angle value in degrees.
    :returns projection: a 2D array in (Y, Z) form.
    """
    projection = radiographs(data, [omega])
    return projection[:, :, 0]

def radiographs(data, omegas):
    """Compute the radiographs of a 3D object using the radon transform.

    The object is represented by a 3D numpy array in (XYZ) form and a series of projection at each omega angle
    are computed assuming the rotation is along Z in the middle of the data set. Internally this function uses
    the radon transform from the skimage package.

    :param np.array data: an array representing the 3D object in (XYZ) form.
    :param omegas: an array of the rotation values in degrees.
    :returns projections: a 3D array in (Y, Z, omega) form.
    """
    assert data.ndim == 3
    if type(omegas) is list:
        omegas = np.array(omegas)
    width = int(np.ceil(max(data.shape[0], data.shape[1]) * 2 ** 0.5))
    projections = np.zeros((width, np.shape(data)[2], len(omegas)), dtype=np.float)
    for z in range(np.shape(data)[2]):
        a = radon(data[:, :, z], -omegas, circle=False)  # - 90  # the 90 seems to come from the radon function itself
        projections[:, z, :] = a[:, :]
    return projections



