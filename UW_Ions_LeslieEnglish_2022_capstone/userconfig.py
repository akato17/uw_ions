# -*- coding: utf-8 -*-
"""
@author: Zeyu Ye, Samip Karki
@desc Introduce the configuration of the trap and the simulation method.
@date 9/3/2021
Modified by Leslie English, June 2022
"""

import math
import numpy as np
from mdconst import *     

################## SET TRAP VOLTAGES #############################

# Set Parameters for Radiofrequency (RF) frequency and amplitude
f = 12.47e6  # RF Frequency (Hz)
omega = np.pi*2*f
period = 1 / f
RFamp=985    # RF Amplitude (V)
print()
print('    Radiofrequency AC voltage settings:')
print('     RF frequency f = ', f)
print('     RF omega = ', omega)
print('     RF amplitude = ', RFamp)
print()
### Set voltage multiplication factors for ring sector elecrodes and end caps
Side_Pinch_V = 0     
Side_Stretch_V = -0   # Must be a negative number so its force goes opposite direction to other electrode voltages.   
Endcap_V = 0
print('    Ring sector & end cap DC voltage multiplication factors:')
print('     Side_Pinch_V = ', Side_Pinch_V)
print('     Side_Stretch_V = ', Side_Stretch_V)
print('     Endcap_V = ', Endcap_V)
print()
 # Assign voltage multiplication factors to specific electrode pairs to create the intended crystal shape & location
Factor04 = Side_Pinch_V     # Multiplication factor electrode pair 0 & 4
Factor15 = 0    # Multiplication factor electrode pair 1 & 5 
Factor26 = 0     # Multiplication factor electrode pair 2 & 6
Factor37 = 0   # Multiplication factor electrode pair 3 & 7
Offset = 0                  # Displacement force to offset crystal from center of trap
print('    DC Voltage multiplication factor assignments to ring sector electrode pairs:')
print('     Factor04 = ', Factor04)
print('     Factor15 = ', Factor15)
print('     Factor26 = ', Factor26)
print('     Factor37 = ', Factor37)
print()


############# SET LASER PARAMETERS ##################

# Define the location (meters), orientation (degrees relative to x-y plane), and cooling laser waist.
# Referencing "A Paul trap with sectored ring electrodes for experiments with two-dimensional ion crystals"
#   by M. K. Ivory, A. Kato, A. Hasanzadeh, & B. B. Blinov, (April 2020) p.2 (doi: 10.1063/1.5145102):
#    TRAP SIZE: "The trap ring sectors are 20° metal wedges with 25° spacing, 
#               with an ID of 4 mm and OD of 10 mm."
# Referencing: "Two-tone Doppler cooling of radial two-dimensional crystals in a radiofrequency ion trap"
#   by Kato, Blinov, et al (November 2021), p.4-5
#    LASER ANGLE: "The beams are focused to a waist of approximately 50 μm, 
#                  making an angle of 0 ~ 10° with respect to the crystal plane."
LaserAngle = 10                        # degrees 
LaserRadians = LaserAngle*2*PI/360     # radians 
StdDev = .00002   # 20 microns = distance (meters) at which laser strength profile is one standard deviation from maximum.


############# SET SIMULATION VARIABLES ################

m = 138*M
kappa = Q**2 / (4 * PI * EPSILON_0)

# initial
start_area = 100e-6
T = 33            # Initial temperature (global variable)

# time steps
basetstep = (1/f)/20 #3.24e-08/3 #tstep if there is only one ion, for discrete laser
tstep= basetstep #need smaller tstep for two step process, for psuedo
time=0

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

### temperature set:
T_set=13e-3
