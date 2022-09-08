# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:31:24 2021
@author: samip
Modified by Leslie English, June 2022
"""
print()
print('STARTING PLOTS')
print()

################## IMPORT PYTHON MODULES, PACKAGES & LIBRARIES #############################

# Module = Single Python file containing functions. 
# Package = Directory of Python modules containing an additional __init__.py file.
import matplotlib.pyplot as plt
import numpy as np                 # Fundamental package for scientific computing in Python

################## IMPORT FUNCTIONS ########################################################

# Functions imported by this file must be stored in the same directory as this file. 
# Function filenames use extension ".py"
# Using "*" in file import means functions calls need not specify they are from that imported file. 
import simfunc as sf       # This file contains the custom functions called by this file.
from userconfig import *   # This file contains all parameters to define the trap (temperature, voltages)
from PyQt5.QtWidgets import QFileDialog   # This file contains the most recent simmain results

########## SELECT & LOAD THE FILE CONTINAING THE SIMULATION DATA TO BE PLOTTED ################

# Use one of these two file identification options. Comment out the option not used.
#
#### OPTION 1 to select file: This command opens a dialog box and lets you select the file ("Run_recent.npz") you want to open.
# Sometimes, the first attempt to run "Run_recent" fails. If that happens, keep the file open and run it a second time.
filename=QFileDialog.getOpenFileName()[0]
#
#### OPTION 2 to select file: This command is manual designation of the file name and its path.
# Manually type the file path and name or copy paste. This is a fix to get around the error with PyQT5.
# filename="/......../Run_recent.npz"
# Load the data file
X=np.load(filename,allow_pickle=True)
N=X['N']    # N is 0-D shape ()
Y=X['Y']    # Y is 2-D shape (6*N, iterations)  Position 0:3N, Velocity 3N:6N
t=X['t']    # t is 1-D shape (iterations)       Time: iterations
D=X['Distances']   # D is 2-D shape (N, iterations)   Distance to Laser 

plt.clf()   # This clears the current/prior plot(s)
#'''
######### PLOT THE <<FINAL>> 3-D CRYSTAL CONFIGURATION (the final 3-D positions of each ion) ##########

xcord=np.zeros(N,dtype=np.float64)
ycord=np.zeros(N,dtype=np.float64)
zcord=np.zeros(N,dtype=np.float64)
for j in range(0,N):
      xcord[j]=Y[3*j,-1]
      ycord[j]=Y[3*j+1,-1]
      zcord[j]=Y[3*j+2,-1]
ax = plt.axes(projection='3d')
ax.set_zlabel(r'Z', fontsize=20)
ax.set_xlim3d(-50e-6, 50e-6)
ax.set_ylim3d(-50e-6,50e-6)
ax.set_zlim3d(-50e-6,50e-6)
ax.scatter3D(xcord, ycord, zcord)
#'''

'''
#### PLOT THE <<INITIAL>> 3-D CRYSTAL CONFIGURATION (the starting 3-D positions of each ion) ####

xcord=np.zeros(N,dtype=np.float64)
ycord=np.zeros(N,dtype=np.float64)
zcord=np.zeros(N,dtype=np.float64)
for j in range(0,N):
      xcord[j]=Y[3*j,0]
      ycord[j]=Y[3*j+1,0]
      zcord[j]=Y[3*j+2,0]
ax = plt.axes(projection='3d')
ax.set_zlabel(r'Z', fontsize=20)
ax.set_xlim3d(-50e-6, 50e-6)
ax.set_ylim3d(-50e-6,50e-6)
ax.set_zlim3d(-50e-6,50e-6)
ax.scatter3D(xcord, ycord, zcord)
'''
'''
#################### PLOT INDIVIDUAL ION TRAJECTORY VS. TIME ##########################

# Identify specific ions to plot

ionnum1=1 #which ion of the N you want to see the plots of. Ion indexing starts at ion 0,1,2,3...
ionnum2=7
print('    1st Ion (ionnum1) is # ' + str(ionnum1) )
print('    2nd Ion (ionnum2) is # ' + str(ionnum2) )
print()

# Within array Y:  Position coordinates are indexes 0 : 3*N. 
#                  Velocity coordinates are indexes 3*N : 6*N.

# ionnum1 position coordinates
x1=Y[3*ionnum1 + 0]
y1=Y[3*ionnum1 + 1]
z1=Y[3*ionnum1 + 2]

# ionnum2 position coordinates
x2=Y[3*ionnum2 + 0]
y2=Y[3*ionnum2 + 1]
z2=Y[3*ionnum2 + 2]

# ionnum1 velocity coordinates
vx1= Y[3*N + 3*ionnum1 + 0]
vy1= Y[3*N + 3*ionnum1 + 1]
vz1= Y[3*N + 3*ionnum1 + 2]

# ionnum2 velocity coordinates
vx2= Y[3*N + 3*ionnum2 + 0]
vy2= Y[3*N + 3*ionnum2 + 1]
vz2= Y[3*N + 3*ionnum2 + 2]

# ionnum1 distance from laser
D1= D[ionnum1]

# ionnum2 distance from laser
D2= D[ionnum2]

# Identify specific data and time range to plot

portion = .005    # Portion [0 to 1] of the simulation time window to plot
'''

'''
# DISTANCE FROM LASER #
plt.plot(  t[0:  int(portion* len(t)) ]  ,  D1[0:int(portion*len(t))]/StdDev, t[0:  int(portion* len(t)) ] , D2[0:int(portion*len(t))]/StdDev  ) # Plot portion from the start.
plt.xlabel('Time (s)')
plt.ylabel('Laser Taper Standard Deviations (sigma)')
plt.ylim((0,6))
plt.title('Distance from Center of Laser Beam for Ion # ' + str(ionnum1) + ' and ' +str(ionnum2))
plt.show()

'''
'''
# POSITIONS #

# X-COORDINATE POSITION
plt.plot(  t[0:  int(portion* len(t)) ]  ,  x1[0:int(portion*len(t))], t[0:  int(portion* len(t)) ] , x2[0:int(portion*len(t))]  ) # Plot portion from the start.
#plt.plot(  t[0:  int(portion* len(t)) ]  ,  x2[0:int(portion*len(x1))]  ) # Plot portion from the start.
#plt.plot(  t[int((1-portion)* len(t)): ]  ,  x1[int((1-portion)*len(x1)):] ) # Plot portion from the end.
plt.xlabel('Time (s)')
plt.ylabel('X-Coordinate of Ion Position')
#plt.title('X-Coordinate Position of Ion # ' + str(ionnum1) )
plt.title('X-Coordinate Position of Ion # ' + str(ionnum1) + ' and ' +str(ionnum2))
plt.show()
'''
'''
# Y-COORDINATE POSITION
plt.plot(  t[0:  int(portion* len(t)) ]  ,  y1[0:int(portion*len(t))], t[0:  int(portion* len(t)) ] , y2[0:int(portion*len(t))]  ) # Plot portion from the start.
#plt.plot(  t[0:  int(portion* len(t)) ]  ,  y2[0:int(portion*len(x1))]  ) # Plot portion from the start.
#plt.plot(  t[int((1-portion)* len(t)): ]  ,  y1[int((1-portion)*len(x1)):] ) # Plot portion from the end.
plt.xlabel('Time (s)')
plt.ylabel('Y-Coordinate of Ion Position')
#plt.title('Y-Coordinate Position of Ion # ' + str(ionnum1) )
plt.title('Y-Coordinate Position of Ion # ' + str(ionnum1) + ' and ' +str(ionnum2))
plt.show()

# Z-COORDINATE POSITION
plt.plot(  t[0:  int(portion* len(t)) ]  ,  z1[0:int(portion*len(t))], t[0:  int(portion* len(t)) ] , z2[0:int(portion*len(t))]  ) # Plot portion from the start.
#plt.plot(  t[0:  int(portion* len(t)) ]  ,  z2[0:int(portion*len(x1))]  ) # Plot portion from the start.
#plt.plot(  t[int((1-portion)* len(t)): ]  ,  z1[int((1-portion)*len(x1)):] ) # Plot portion from the end.
plt.xlabel('Time (s)')
plt.ylabel('Z-Coordinate of Ion Position')
#plt.title('Z-Coordinate Position of Ion # ' + str(ionnum1) )
plt.title('Z-Coordinate Position of Ion # ' + str(ionnum1) + ' and ' +str(ionnum2))
plt.show()
'''
#  VELOCITIES  #
'''
'#''
# X-COORDINATE VELOCITY
plt.plot(  t[0:  int(portion* len(t)) ]  ,  vx1[0:int(portion*len(t))], t[0:  int(portion* len(t)) ] , vx2[0:int(portion*len(t))]  ) # Plot portion from the start.
#plt.plot(  t[0:  int(portion* len(t)) ]  ,  vx2[0:int(portion*len(x1))]  ) # Plot portion from the start.
#plt.plot(  t[int((1-portion)* len(t)): ]  ,  vx1[int((1-portion)*len(x1)):] ) # Plot portion from the end.
plt.xlabel('Time (s)')
plt.ylabel('X-Coordinate of Ion Velocity')
#plt.title('X-Coordinate Velocity of Ion # ' + str(ionnum1) )
plt.title('X-Coordinate Velocity of Ion # ' + str(ionnum1) + ' and ' +str(ionnum2))
plt.show()
'''
'''
# Y-COORDINATE VELOCITY
plt.plot(  t[0:  int(portion* len(t)) ]  ,  vy1[0:int(portion*len(t))], t[0:  int(portion* len(t)) ] , vy2[0:int(portion*len(t))]  ) # Plot portion from the start.
#plt.plot(  t[0:  int(portion* len(t)) ]  ,  vy2[0:int(portion*len(x1))]  ) # Plot portion from the start.
#plt.plot(  t[int((1-portion)* len(t)): ]  ,  vy1[int((1-portion)*len(x1)):] ) # Plot portion from the end.
plt.xlabel('Time (s)')
plt.ylabel('Y-Coordinate of Ion Velocity')
#plt.title('Y-Coordinate Velocity of Ion # ' + str(ionnum1) )
plt.title('Y-Coordinate Velocity of Ion # ' + str(ionnum1) + ' and ' +str(ionnum2))
plt.show()
'''
'''
# Z-COORDINATE VELOCITY
plt.plot(  t[0:  int(portion* len(t)) ]  ,  vz1[0:int(portion*len(t))], t[0:  int(portion* len(t)) ] , vz2[0:int(portion*len(t))]  ) # Plot portion from the start.
#plt.plot(  t[0:  int(portion* len(t)) ]  ,  vz2[0:int(portion*len(x1))]  ) # Plot portion from the start.
#plt.plot(  t[int((1-portion)* len(t)): ]  ,  vz1[int((1-portion)*len(x1)):] ) # Plot portion from the end.
plt.xlabel('Time (s)')
plt.ylabel('Z-Coordinate of Ion Velocity')
#plt.title('Z-Coordinate Velocity of Ion # ' + str(ionnum1) )
plt.title('Z-Coordinate Velocity of Ion # ' + str(ionnum1) + ' and ' +str(ionnum2))
plt.show()
'''

print("PLOTTING COMPLETE.")
print()