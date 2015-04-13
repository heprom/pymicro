'''The fitting module define some classes to easily perform 1D curve 
   fitting.
'''
from scipy import optimize
from scipy.special import wofz
import numpy as np

def fit(y, x = None, fit_type='Gaussian'):
  '''Static method to perform curve fitting directly.

  For instance, to fit some (x,y) data with a gaussian function, simply use:
  ::

    F = fit(y, x, fit_type='Gaussian')

  '''
  if fit_type == 'Gaussian':
    F = Gaussian()
  elif fit_type == 'Lorentzian':
    F = Lorentzian()
  elif fit_type == 'Cosine':
    F = Cosine()
  else:
    print('unknown fitting type: %s' % fit_type)
    return None
  F.fit(y, x)
  return F

class Parameter:
  '''A class to handle modiable parameters.'''
  
  def __init__(self, value, name = None):
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

  def fit(self, y, x = None, verbose = False):
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
          print 'iteration %d, trying parameters:' % it, p
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
        for i, pi in enumerate(p):
          pi.set(new_params[i])
        return y - self(x)
        
      if x is None: x = np.arange(y.shape[0])
      p = [param.value for param in self.get_parameters()]
      optimize.leastsq(cost_func, p, Dfun= None)

class Gaussian(FitFunction):
  '''first parameter is position, second is sigma, third is height'''
  def __init__(self, position=0.0, sigma=1.0, height=1.0):
    FitFunction.__init__(self)
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
    return 2*p[1].value*np.log(2)
  
  def compute(self, x):
    '''Evaluate the gauss function at coordinates x.'''
    p = self.get_parameters()
    return p[2].value * np.exp(-((x-p[0].value)/p[1].value)**2)

class Lorentzian(FitFunction):
  '''first parameter is position, second is gamma'''
  def __init__(self, position=0.0, gamma=1.0):
    FitFunction.__init__(self)
    self.add_parameter(position, 'position')
    self.add_parameter(gamma, 'width')
  
  def set_position(self, position):
    self.parameters[0].set(position)
    
  def set_gamma(self, gamma):
    self.parameters[1].set(gamma)

  def fwhm(self):
    p = self.get_parameters()
    return 2*p[1].value

  def compute(self, x):
    p = self.get_parameters()
    return p[1].value/np.pi/((x-p[0].value)**2 + p[1].value**2)

class Cosine(FitFunction):
  '''first parameter is position, second is width'''
  def __init__(self, position=0.0, width=1.0):
    FitFunction.__init__(self)
    self.add_parameter(position, 'position')
    self.add_parameter(width, 'width')
  
  def set_position(self, position):
    self.parameters[0].set(position)
    
  def set_width(self, a):
    self.parameters[1].set(a)

  def fwhm(self):
    p = self.get_parameters()
    return 3./4*p[1].value

  def compute(self, x):
    p = self.get_parameters()
    return np.cos(np.pi*(x-p[0].value)/(2*p[1].value))

def voigt(x, sigma, gamma):
   # The Voigt function is also the real part of 
   # w(z) = exp(-z^2) erfc(iz), the complex probability function,
   # which is also known as the Faddeeva function. Scipy has 
   # implemented this function under the name wofz()
   z = (x + 1j*gamma) / (sigma * np.sqrt(2))
   I = wofz(z).real / (sigma * np.sqrt(2) * np.pi)
   return I
