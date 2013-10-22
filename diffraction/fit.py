import numpy

class fitFunc():
  (linear, pearson7, pearson7asym) = range(3)
  types = ['linear', 'pearson7', 'pearson7asym']
  npar = [2, 4, 6]
  par_labels = [['slope', 'yorig'], \
    ['peak height', 'peak position', 'half height width', 'shape factor'], \
    ['peak height', 'peak position', 'left half height width', 'left shape factor', 'right half height width', 'right shape factor'], \
    ]
  par_units = [['1/degree', '-'], \
    ['-', 'degree', 'degree', '-'], \
    ['-', 'degree', 'degree', '-', 'degree', '-'], \
    ]

  def __init__(self, name='no_name', type=0, params=[0., 0.]):
    self.name = name
    # check type
    index = 0
    try:
      index = range(3).index(type)
      #index = fitFunc.types.index(type)
    except ValueError:
      print 'invalid type:',type,'falling back on linear...'
      type = fitFunc.linear
    print 'init new fitfunc with type', type, 'index is',index
    self.type = type
    self.plot = False
    # check params
    print 'params len=',len(params)
    if not (len(params) == fitFunc.npar[self.type]):
      print 'WARNING: number of parameters for function of type', self.types[self.type]
      print 'falling back on default values...'
      params = numpy.ones(fitFunc.npar[self.type], dtype=numpy.float).tolist()
    print 'params=',params
    self.params = params
  
  def __str__(self):
    return 'fitFunc of type ' + fitFunc.types[self.type] + ' with params ' + str(self.params)
  
  def compute(self, x):
    if self.type == fitFunc.linear:
      return float(self.params[0])*x + float(self.params[1])
    elif self.type == fitFunc.pearson7:
      I0 = self.params[0] # peak height
      x0 = self.params[1] # peak position
      d  = self.params[2] # half width height
      m  = self.params[3] # shape factor
      return I0/(1.+4.*(2.**(1./m)-1.)*((x-x0)/d)**2.)**m
    elif self.type == fitFunc.pearson7asym:
      I0 = self.params[0] # peak height
      x0 = self.params[1] # peak position
      dl = self.params[2] # left half width height
      ml = self.params[3] # left shape factor
      dr = self.params[4] # right half width height
      mr = self.params[5] # right shape factor
      I = numpy.zeros_like(x)
      for i in range(len(x)):
        if x[i] <= x0:
          I[i] = I0/(1.+4.*(2.**(1./ml)-1.)*((x[i]-x0)/dl)**2.)**ml
        else:
          I[i] = I0/(1.+4.*(2.**(1./mr)-1.)*((x[i]-x0)/dr)**2.)**mr
      return I

if __name__ == '__main__':
  func = fitFunc(name='toto')
  print func

