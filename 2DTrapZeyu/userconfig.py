# -*- coding: utf-8 -*-
"""
@author: Zeyu Ye
@desc Introduce the configuration of the trap and the simulation method.
@date 9/16/2020
"""

import numpy as np
from mdconst import *

m = 138*M
kappa = Q**2 / (4 * PI * EPSILON_0)

# initial
start_area = 200e-6
T = 150

# simulation
#twindow = 5e-5
twindow = 5e-2
tstep = 3.24e-08
#tstep = 4e-9

# grid properties
gmax = .005 # max and min grid point position
size = 401 # number of grid points

# fitting properties
n_points = 21 # number of points for fitting
grid = gmax * 2 / size * n_points

# virtual gas
m_g=m/100
T_g=2e-3
mu_g=np.sqrt(8* K *T_g/(PI*m_g)) # mean
sigma_g=np.sqrt(K*T_g/(2*m_g)) # std

#trap parameters
f = 12.08*(10**6)
omega = np.pi*2*f
period = 1 / f

field_config = {'half_grid_points': 200, 
                'half_grid_length': 0.005}