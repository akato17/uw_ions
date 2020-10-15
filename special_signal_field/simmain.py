# -*- coding: utf-8 -*-
"""
@author: Zeyu Ye
@desc An example of using the functions.
@date 10/15/2020
"""

import numpy as np
import fieldfunc as ff
import simfunc as sf
import matplotlib.pyplot as plt
from userconfig import *

# Input the signal as array of sampled points
signal = np.cos(np.linspace(0,2*np.pi,sample_points + 1))[:-1] * 1000

# Create the V_0 field
RF=np.array([0,0,0,0,0,0,0,0,1000,1000])
DC=np.array([30,0,0,0,30,0,0,0,0,0])
V0RF, V0DC, cord = ff.trap_potentials(RF, DC)
paramsRF, paramsDC = ff.curve_fit(V0RF, V0DC, cord, 1000, 30)

# Load Fields from files
#paramsRF = np.load('1000VparamsRF.npy')
#paramsDC = np.load('paramsDC30.0.npy')

# Run simulation
N = 13
IC = sf.initialize_ions(N)
P = sf.leap_frog(N,IC,paramsRF,paramsDC,signal)

# Plot
xcord=np.zeros(N,dtype=np.float64)
ycord=np.zeros(N,dtype=np.float64)
zcord=np.zeros(N,dtype=np.float64)

for i in range(0,N):
      xcord[i]=P[1][3*i,-1]
      ycord[i]=P[1][3*i+1,-1]
      zcord[i]=P[1][3*i+2,-1]

ax = plt.axes(projection='3d')
ax.set_zlabel(r'Z', fontsize=20)
ax.scatter3D(xcord, ycord, zcord)

# Save Result
np.save('P.npy',(P[0],P[1]))