"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pymicro.view.vol_utils import format_coord

X = 10*np.random.rand(50, 30)

fig, ax = plt.subplots()
ax.imshow(X, cmap=cm.jet, interpolation='nearest')
ax.format_coord = format_coord
plt.show()

