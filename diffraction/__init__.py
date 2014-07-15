"""
    Waxd
    ~~~~
    #----------------------------------------------------------------------
    # Name:        Waxd
    # Purpose:     A package to analyze experimental WAXD data
    #
    # Author:      Henry Proudhon
    #
    # Created:     30-Mar-2009
    # Copyright:   (c) 2009-2010
    # License:     GPLv3 license
    #----------------------------------------------------------------------
    .. moduleauthor:: Henry Proudhon <henry.proudhon@mines-paristech.fr>
"""
__author__  = "Henry Proudhon <henry.proudhon@mines-paristech.fr>"
__date__    = "30 Jan 2010, 15:45 GMT+01:00"
__version__ = "0.21"
__doc__     = """\
package providing tools to analyze experimental WAXD patterns, including radial and phi profiles and fitting functions to isolate cristallization peaks.
"""

# import relevant external symbols into package namespace:
from pymicro.diffraction.waxd import *
from pymicro.diffraction.fit import *
