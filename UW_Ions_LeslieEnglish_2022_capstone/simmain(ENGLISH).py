# -*- coding: utf-8 -*-
"""
@author: Samip
@desc An example of using the functions.
@date 9/3/2021
Modified by Leslie English, June 2022
"""
print()
print('STARTING SIMULATION')
print()
#
################## IMPORT PYTHON MODULES, PACKAGES & LIBRARIES #############################
# Module = Single Python file containing functions. 
# Package = Directory of Python modules containing an additional __init__.py file.
import matplotlib.pyplot as plt
import numpy as np                 # Fundamental package for scientific computing in Python
#
################## IMPORT FUNCTIONS ########################################################
# Functions imported by this file must be stored in the same directory as this file. 
# Function filenames use extension ".py"
# Using "*" in file import means functions calls and variables need not specify they are from that imported file.
import simfunc as sf       # This file contains most (all???) functions called by this file.
from userconfig import *   # This file contains all paramenters to define the trap (temperature, voltages)
from PyQt5.QtWidgets import QFileDialog   # This dialog box opens the file that contains the most recent simmain results

### SIMULATION STEP 1: Define number of ions in the trap.
N = 25      # N type =  <class 'int'>

### SIMULATION STEP 2: Initialize conditions in the trap.
# (2(a)) Initialization OPTION 1: Call the "initialize_ions" function defined in "simfunc" file.
IC = sf.initialize_ions(N)     # IC is a (1-D) array of shape (6*N,). Type =  <class 'numpy.ndarray'>
# (2(b)) Initialization OPTION 2:  # OR start with another file
# load = np.load('54_ions.npz',allow_pickle=True)#[1]
# IC = np.zeros(6*N)
accf=np.zeros(3*N) # At this point this is actually the starting acceleration, not final acceleration.
tstart=0
twindow=5e-3  # seconds

### SIMULATION STEP 3: Run Leap Frog Numerical Integration Method
P= sf.leap_frog3(N, IC, Side_Stretch_V, Side_Pinch_V, Endcap_V, RFamp, tstart, twindow, accf) # TYPE P =  <class 'tuple'>
t=P[0]      #  Time. Array (1-D) = twindow/tstep elements  (ie: number of iterations). TYPE =  <class 'numpy.ndarray'>
Y=P[1]      #  Position & Velocity. Array (2-D) = 6N rows (x,y,z,vx,vy,vz for N ions) x twindow/tstep columns. TYPE =  <class 'numpy.ndarray'> 
accf=P[2]   #  Final Acceleration (last time increment). Array size (1-D) = 3N (ax, ay, az for each ion). TYPE =  <class 'numpy.ndarray'>
Distances=P[3] # (2-D) Array. (N x iterations). Distance of each ion from center of the later beam.

### SIMULATION STEP 4: Save data from completed simulation to "Run_recent.npz" file
# Changing data type to float 32 to save space.
t=np.array(t,dtype=np.float32)
Y=np.array(Y,dtype=np.float32) 
accf=np.array(accf,dtype=np.float32)
Distances=np.array(Distances,dtype=np.float32)

# Saving data.
np.savez('Run_recent', t=t, Y=Y, N=N, accf=accf, Distances=Distances)

print('SIMULATION IS COMPLETE in "simmain.py": Data results have been stored in files "Run_recent.npz".')
print()
print('NEXT STEP IS TO PLOT THE RESULTS: Open and run "plotsim.py". In the dialog box, select "Run_recent.npz." ', end = '')
print('If the dialog box will not allow you to select the file, try re-running plotsim. The second attempt usually works.')
print()
