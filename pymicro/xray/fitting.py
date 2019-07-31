'''The fitting module define some classes to easily perform 1D curve 
   fitting. This module supports fitting (x, y) data with a general 
   mechanism. Any fitting function can be provided to the fit method 
   and a few general purpose fuctions are predefined:
   
    * Gaussian
    * Lorentzian
    * Cosine
    * Voigt

  .. figure:: _static/fitting_functions.png
      :width: 400 px
      :height: 300 px
      :alt: lattice_3d
      :align: center

      Plot of the main predefined fitting function in the fitting module.
'''
from scipy import optimize
from scipy.special import wofz
import numpy as np


def fit(y, x=None, expression=None, nb_params=None, init=None):
    '''Static method to perform curve fitting directly.

      *Parameters*

      **y**: the data to match (a 1d numpy array)

      **x**: the corresponding x coordinates (optional, None by default)

      **expression**: can be either a string to select a predefined
      function or alternatively a user defined function with the signature
      f(x, p) (in this case you must specify the length of the parameters
      array p via setting nb_params).

      **nb_params**: the number of parameters of the user defined fitting
      function (only needed when a custom fitting function is provided,
      None by default)

      **init**: a sequence (the length must be equal to the number of
      parameters of the fitting function) used to initialise the fitting
      function.

      For instance, to fit some (x,y) data with a gaussian function, simply use:
       ::

         F = fit(y, x, expression='Gaussian')

       Alternatively you may specify you own function directly defined with Python, like:
       ::

         def myf(x, p):
           return p[0]*x + p[1]

         F = fit(y, x, expression=myf, nb_params=2)
    '''
    if expression == 'Gaussian':
        F = Gaussian()
    elif expression == 'Lorentzian':
        F = Lorentzian()
    elif expression == 'Cosine':
        F = Cosine()
    elif expression == 'Voigt':
        F = Voigt()
    else:
        F = FitFunction()
        if not nb_params:
            print('please specify the number of parameters for your fit function, aborting fit...')
            return None
        if not init:
            init = np.ones(nb_params)
        if not len(init) == nb_params:
            print(
                'there are more parameters in the fit function than specified in the initialization sequence, aborting initialization...')
            init = np.ones(nb_params)
        for i in range(nb_params):
            F.add_parameter(init[i], 'p%d' % i)
        F.expression = expression
    F.fit(y, x)
    return F


def lin_reg(xi, yi):
    """Apply linear regression to a series of points.
    
    This function return the best linear fit in the least square sense.
    :param ndarray xi: a 1D array of the x coordinate.
    :param ndarray yi: a 1D array of the y coordinate.
    :return tuple: the linear intercept, slope and correlation coefficient.
    """
    n = len(xi)
    assert (n == len(yi))
    sx = np.sum(xi)
    sy = np.sum(yi)
    sxx = np.sum(xi ** 2)
    sxy = np.sum(xi * yi)
    syy = np.sum(yi ** 2)
    beta = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    alpha = 1. / n * sy - 1. / n * sx * beta
    r = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    return alpha, beta, r


class Parameter:
    '''A class to handle modiable parameters.'''

    def __init__(self, value, name=None):
        '''Create a new parameter with the given value.'''
        self.value = value
        self.name = name

    def set(self, value):
        '''Set the value of the parameter.'''
        self.value = value

    def set_name(self, name):
        '''Set the name of the parameter.'''
        self.name = name

    def __call__(self):
        '''With this we can use p() if p is an instance of Parameter.'''
        return self.value

    def __repr__(self):
        '''Provide a string representation of the parameter, simply based
        on its actual value.'''
        return str(self.value)


class FitFunction:
    '''This class provides a basic canvas to define a fit function.

    You may subclass it to create your own fitting function just as the
    predfined fit function do (see `Gaussian` for instance).
    '''

    def __init__(self):
        self.parameters = []
        self.expression = None

    def get_parameters(self):
        '''Return the list of parameters of this fit function.'''
        return self.parameters

    def get_parameter_names(self):
        '''Return the list os parameter names of this fit function.'''
        names = []
        for p in self.get_parameters:
            names.append(p.name)
        return names

    def add_parameter(self, value, name):
        param = Parameter(value, name)
        self.parameters.append(param)

    def __call__(self, x):
        '''With this we can call directly f(x) to evaluate the function f at x.'''
        return self.compute(x)

    def __repr__(self):
        '''Provide a string representation of the fitting function, giving
        its type and the list of its parameters with names and values.'''
        s = '%s function\n' % self.__class__.__name__
        s += 'Parameters are:\n'
        params = self.get_parameters()
        for i in range(len(params)):
            s += ' * %s = %g\n' % (params[i].name, params[i].value)
        return s

    def compute(self, x):
        '''Evaluate the fit function at coordinates x.'''
        p = self.get_parameters()
        return self.expression(x, p)

    def fit(self, y, x=None, verbose=False):
        '''Perform fitting on the given data.

        This will adjust the parameters of this fit function to match as
        well as possible the given data using a least square minimisation.

        *Parameters*

        **y**: the data to match (a 1d numpy array)

        **x**: the corresponding x coordinates (optional, None by default)

        **verbose**: boolean, activate verbose mode
        '''
        # local function to evaluate the cost
        global it
        it = 0

        def cost_func(new_params):
            global it
            p = self.get_parameters()
            if verbose:
                print('iteration %d, trying parameters:' % it, p)
                it += 1
                '''
                from matplotlib import pyplot as plt
                if it == 0:
                  plt.plot(x, y, 'bo', label = 'data points')
                  plt.ylim(0, 0.6)
                  plt.grid()
                  plt.legend(numpoints=1,loc='upper left')
                  plt.savefig('fit/fit_%02d.pdf' % it)
                it += 1
                plt.clf()
                plt.plot(x, y, 'bo', label = 'data points')
                plt.plot(x, self(x), 'k-', label = 'gaussian fit')
                plt.ylim(0, 0.6)
                plt.title('fitting iteration %02d' % it)
                plt.legend(numpoints=1,loc='upper left')
                plt.savefig('fit/fit_%02d.pdf' % it)
                '''
            for i, pi in enumerate(p):
                pi.set(new_params[i])
            return y - self(x)

        if x is None: x = np.arange(y.shape[0])
        p = [param.value for param in self.get_parameters()]
        optimize.leastsq(cost_func, p, Dfun=None, xtol=1.e-6)

class SumOfFitFunction(FitFunction):

    def __init__(self, function_list):
        self.parameters = []
        for f in function_list:
            self.parameters.extend(f.get_parameters())
        self.expression = function_list

    def __repr__(self):
        '''Provide a string representation of the fitting function, giving
        its type and the list of its parameters with names and values.'''
        s = '%s function\n' % self.__class__.__name__
        s += 'list of function in the sum:'
        for f in self.expression:
            s += f.__repr__()
        return s

    def compute(self, x):
        '''Evaluate the fit function at coordinates x.'''
        result = np.zeros_like(x)
        for f in self.expression:
            result += f.compute(x)
        return result


class Gaussian(FitFunction):
    '''first parameter is position, second is sigma, third is height'''

    def __init__(self, position=0.0, sigma=1.0, height=1.0):
        FitFunction.__init__(self)

        def G(x, p):
            return p[2].value * np.exp(-((x - p[0].value) / p[1].value) ** 2)

        self.expression = G
        self.add_parameter(position, 'position')
        self.add_parameter(sigma, 'sigma')
        self.add_parameter(height, 'height')

    def set_position(self, position):
        '''Set the position (center) of the gauss function.'''
        self.parameters[0].set(position)

    def set_sigma(self, sigma):
        '''Set the width (variance) of the gauss function.'''
        self.parameters[1].set(sigma)

    def set_height(self, height):
        '''Set the maximum (height) of the gauss function.'''
        self.parameters[2].set(height)

    def fwhm(self):
        '''Compute the full width at half maximum of the gauss function.'''
        p = self.get_parameters()
        return 2 * p[1].value * np.sqrt(np.log(2))


class Lorentzian(FitFunction):
    '''Lorentzian funtion.

    The first parameter is the position, the second is gamma. The maximum
    of the function is given by height_factor/(pi*gamma). The FWHM is just 2*gamma.
    '''

    def __init__(self, position=0.0, gamma=1.0, height_factor=1.0):
        FitFunction.__init__(self)

        def L(x, p):
            return p[2].value * p[1].value / np.pi / ((x - p[0].value) ** 2 + p[1].value ** 2)

        self.expression = L
        self.add_parameter(position, 'position')
        self.add_parameter(gamma, 'width')
        self.add_parameter(height_factor, 'height_factor')

    def set_position(self, position):
        self.parameters[0].set(position)

    def set_gamma(self, gamma):
        self.parameters[1].set(gamma)

    def set_height(self, height):
        '''Set the maximum (height) of the Lorentzian function. This
        actually set the height factor to the value height*pi*gamma.
        '''
        self.parameters[2].set(height * np.pi * self.parameters[1].value)

    def fwhm(self):
        p = self.get_parameters()
        return 2 * p[1].value


class Cosine(FitFunction):
    '''first parameter is position, second is width'''

    def __init__(self, position=0.0, width=1.0):
        FitFunction.__init__(self)

        def C(x, p):
            return np.cos(np.pi * (x - p[0].value) / (2 * p[1].value))

        self.expression = C
        self.add_parameter(position, 'position')
        self.add_parameter(width, 'width')

    def set_position(self, position):
        self.parameters[0].set(position)

    def set_width(self, a):
        self.parameters[1].set(a)

    def fwhm(self):
        p = self.get_parameters()
        return 3. / 4 * p[1].value


class Voigt(FitFunction):
    '''The Voigt function is also the real part of
    w(x) = exp(-x**2) erfc(ix), the Faddeeva function.

    Here we use one of the popular implementation which is available
    in scipy with the wofz function.
    '''

    def __init__(self, position=0.0, sigma=1.0, gamma=1.0, height_factor=1.0):
        FitFunction.__init__(self)

        def V(x, p):
            z = (x - p[0].value + 1j * p[2].value) / (p[1].value * np.sqrt(2))
            return p[3].value * wofz(z).real / (p[1].value * np.sqrt(2 * np.pi))

        self.expression = V
        self.add_parameter(position, 'position')
        self.add_parameter(sigma, 'sigma')
        self.add_parameter(gamma, 'gamma')
        self.add_parameter(height_factor, 'height_factor')

    def set_position(self, position):
        '''Set the position (center) of the Voigt function.'''
        self.parameters[0].set(position)

    def set_sigma(self, sigma):
        '''Set the sigma of the Voigt function.'''
        self.parameters[1].set(sigma)

    def set_height(self, height):
        '''Set the maximum (height) of the Voigt function. This
        actually set the height factor to the proper value. Be careful that
        if you change the other parameters (sigma, gamma) the maximum height
        will be changed.
        '''
        maxi = self.compute(self.parameters[0].value)
        self.parameters[3].set(height / maxi)

    def fwhm(self):
        '''Compute the full width at half maximum of the Voigt function.

        The height factor does not change the fwhm. The fwhm can be evaluated
        by the width of the associated Gaussian and Lorentzian functions.
        The fwhm is approximated by the equation from J. Olivero and R. Longbothum [1977]'''
        p = self.get_parameters()
        fg = 2 * p[1].value * np.sqrt(2 * np.log(2))
        fl = 2 * p[2].value
        return 0.5346 * fl + np.sqrt(0.2166 * fl ** 2 + fg ** 2)
