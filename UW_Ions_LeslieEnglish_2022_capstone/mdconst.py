# -*- coding: utf-8 -*-
"""
@author: Alex Kato
@desc Introduce all simulation related constants for import.
@date 9/16/2020
"""

'''SciPy provides algorithms for optimization, integration, interpolation, eigenvalue problems, algebraic equations, 
differential equations, statistics and many other classes of problems.'''
import scipy.constants as sc
''' NumPy offers Numerical computing tools: comprehensive mathematical functions, random number generators, 
linear algebra routines, Fourier transforms, and more. N-Dimensional Arrays. '''
import numpy as np

from scipy.constants import e as Q
from scipy.constants import k as K
from scipy.constants import epsilon_0 as EPSILON_0
from numpy import pi as PI

from scipy.constants import c as C
from scipy.constants import hbar
from scipy.constants import h as H
from scipy.constants import physical_constants

M=physical_constants['atomic mass constant'][0]
