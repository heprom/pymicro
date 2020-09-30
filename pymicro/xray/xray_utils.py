import os, numpy as np
from matplotlib import pyplot as plt
from skimage.transform import radon
from math import *
from config import PYMICRO_XRAY_DATA_DIR

class Element:
    """A class to represent a chemical element described by its name, symbol and density."""

    def __init__(self, name, symbol, density):
        self._name = name
        self._symbol = symbol
        self._density = density

    @property
    def name(self):
        """Returns the name of the `Element`."""
        return self._name

    @property
    def symbol(self):
        """Returns the symbol of the `Element`."""
        return self._symbol

    @property
    def density(self):
        """Returns the density of the `Element`."""
        return self._density

    def __repr__(self):
        return '%s, %s, density %.3f g.cm^-3' % (self.symbol, self.name, self.density)


elements = {3: Element('Lithium', 'Li', 0.533),
            4: Element('Berylium', 'Be', 1.8450),
            6: Element('Carbon', 'C', 2.26),
            11: Element('Sodium', 'Na', 0.968),
            12: Element('Magnesium', 'Mg', 1.738),
            13: Element('Aluminium', 'Al', 2.6941),
            14: Element('Silicium', 'Si', 2.33),
            22: Element('Titanium', 'Ti', 4.530),
            23: Element('Vanadium', 'V', 6.100),
            24: Element('Chromium', 'Cr', 7.180),
            25: Element('Manganese', 'Mn', 7.430),
            26: Element('Iron', 'Fe', 7.874),
            27: Element('Cobalt', 'Co', 8.900),
            28: Element('Nickel', 'Ni', 8.902),
            29: Element('Copper', 'Cu', 8.960),
            30: Element('Zinc', 'Zn', 7.112),
            31: Element('Gallium', 'Ga', 7.877),
            32: Element('Germanium', 'Ge', 5.370),
            33: Element('Arsenic', 'As', 5.73),
            34: Element('Selenium', 'Se', 4.79),
            38: Element('Strontium', 'Sr', 2.54),
            39: Element('Yttrium', 'Y', 4.472),
            40: Element('Zirconium', 'Zr', 6.52),
            41: Element('Niobium', 'Nb', 8.550),
            42: Element('Molybdenum', 'Mo', 10.28),
            43: Element('Technetium', 'Tc', 11.50),
            47: Element('Silver', 'Ag', 10.49),
            48: Element('Cadmium', 'Cd', 8.65),
            49: Element('Indium', 'In', 7.31),
            50: Element('Tin', 'Sn', 7.265),
            74: Element('Tungsten', 'W', 19.3),
            82: Element('Lead', 'Pb', 11.330),
            83: Element('Bismuth', 'Bi', 9.747)
            }


def density_from_Z(Z):
    if not Z in elements:
        print('unknown atomic number: %d' % Z)
        return None
    return elements[Z].density


def density_from_symbol(symbol):
    for Z in elements:
        if elements[Z].symbol == symbol:
            return elements[Z].density
    print('unknown symbol: %s' % symbol)
    return None


def f_atom(q, Z):
    """Empirical function for the atomic form factor f.

    This function implements the empirical function from the paper of Muhammad and Lee
    to provide value of f for elements in the range Z <= 30.
    doi:10.1371/journal.pone.0069608.t001

    :param float q: value or series for the momentum transfer, unit is angstrom^-1
    :param int Z: atomic number, must be lower than 30.
    :return: the atomic form factor value or series corresponding to q.
    """
    if Z > 30:
        raise ValueError('only atoms with Z<=30 are supported, consider using tabulated data.')
    a0 = 0.52978  # angstrom, Bohr radius
    params = np.genfromtxt(os.path.join(PYMICRO_XRAY_DATA_DIR, 'f_atom_params.txt'), names=True)
    Z, r, a1, b1, a2, b2, a3, b3, _, _, _ = params[int(Z - 1)]
    print(Z, r, a1, b1, a2, b2, a3, b3)
    f = (a1 * Z) ** r / ((a1 * Z) ** r + b1 * (2 * pi * a0 * q) ** r) ** r + \
        (a2 * Z) ** r / ((2 * a2 * Z) ** r + b2 * (2 * pi * a0 * q) ** 2) ** 2 + \
        (a3 * Z) ** r / ((2 * a3 * Z) ** 2 + b3 * (2 * pi * a0 * q) ** 2) ** 2
    return f


def atom_scat_factor_function(mat='Al', sintheta_lambda_max=12, display=True):
    """Compute and display the fit function of the atomic scattering factor.

    :param string mat: A string representing the material (e.g. 'Al')
    :param float sintheta_lambda_max: maximal value of sin theta / lambda
    :param bool display: display an image of the plot
    """
    param = np.genfromtxt(os.path.join(PYMICRO_XRAY_DATA_DIR, mat + '_fit_fatom'))
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
    """Change the unit of wavelength from keV to nm.

    :param float lambda_keV: the wavelength in keV unit.
    :returns: the wavelength in nm unit.
    """
    return 1.2398 / lambda_keV


def lambda_keV_to_angstrom(lambda_keV):
    """Change the unit of wavelength from keV to angstrom.

    :param float lambda_keV: the wavelength in keV unit.
    :returns: the wavelength in angstrom unit.
    """
    return 12.398 / lambda_keV


def lambda_nm_to_keV(lambda_nm):
    """Change the unit of wavelength from nm to keV.

    :param float lambda_nm: the wavelength in nm unit.
    :returns: the wavelength in keV unit.
    """
    return 1.2398 / lambda_nm


def lambda_angstrom_to_keV(lambda_angstrom):
    """Change the unit of wavelength from angstrom to keV.

    :param float lambda_angstrom: the wavelength in angstrom unit.
    :returns: the wavelength in keV unit.
    """
    return 12.398 / lambda_angstrom


def plot_xray_trans(mat='Al', ts=[1.0], rho=None, energy_lim=[1, 100], legfmt='%.1f', display=True):
    """Plot the transmitted intensity of a X-ray beam through a given material.

    This function compute the transmitted intensity from tabulated data of
    the mass attenuation coefficient \mu_\rho (between 1 and 100 keV) and
    applying Beer's Lambert law:

    .. math::

      I/I_0 = \exp(-\mu_\rho*\rho*t)

    The tabulated data is stored in ascii files in the data folder. It has been retrieved
    from NIST `http://physics.nist.gov/cgi-bin/ffast/ffast.pl`
    The density is also tabulated and can be left blanked unless a specific value is to be used.

    :param string mat: a string representing the material (must be the atomic symbol if the density is not specified, e.g. 'Al')
    :param list ts: a list of thickness values of the material in mm ([1.0] by default)
    :param float rho: density of the material in g/cm^3 (None by default)
    :param list energy_lim: energy bounds in keV for the plot (1, 100 by default)
    :param string legfmt: string to format the legend plot
    :param bool display: display or save an image of the plot (False by default)
    """
    mu_rho = np.genfromtxt(os.path.join(PYMICRO_XRAY_DATA_DIR, mat + '.txt'), usecols=(0, 1), comments='#')
    print('Data :', mu_rho)
    energy = mu_rho[:, 0]
    print('Energy :', energy)
    # look up density
    if rho is None:
        rho = density_from_symbol(mat)
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



