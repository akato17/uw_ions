# -*- coding: utf-8 -*-
"""
@author: Zeyu Ye, Samip Karki
@desc Introduce the configuration of the trap and the simulation method.
@date 9/3/2021
"""

import math
import numpy as np
from mdconst import *

m = 138*M
kappa = Q**2 / (4 * PI * EPSILON_0)

# initial
start_area = 200e-6
T = 150

# simulation
#twindow = 5e-5
# twindow = 5e-2
twindow = 5e-3
#tstep = 1.21e-08#check this is the same as the one in laser
basetstep = 3.24e-08/3 #tstep if there is only one ion, for discrete laser
tstep= basetstep #need smaller tstep for two step process, for psuedo
#tstep= basetstep/12 #more that 100x smaller than rf freq,for fr potential
#tstep= 3.311e-09 #1/25 of rf frequency
#tstep= 8.1e-8
#tstep= 8.278e-10 #1/100 of rf frequency
time=0
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