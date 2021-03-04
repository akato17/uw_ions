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
import scipy.integrate as INT
from Field import Field
from userconfig import *
import laser

real_field = Field(mode = 'import')
#real_field.save_params()

# Run simulation
N = 8
#IC = sf.initialize_ions(N)
load = np.load('8ion.npy',allow_pickle=True)[1]
IC = np.zeros(6*N)
for i in range(0,len(load)):
      IC[i] = load[i][-1]
IC = np.concatenate((IC,np.zeros(N)))
P = INT.solve_ivp(lambda t, y: sf.Newton4(t,y,N, real_field.paramsDC, real_field.paramsRF), [0,twindow], y0=IC, t_eval=np.arange(0, twindow, tstep), method='RK45',max_step=tstep)

# Plot
xcord=np.zeros(N,dtype=np.float64)
ycord=np.zeros(N,dtype=np.float64)
zcord=np.zeros(N,dtype=np.float64)

for i in range(0,N):
      xcord[i]=P.y[3*i,-1]
      ycord[i]=P.y[3*i+1,-1]
      zcord[i]=P.y[3*i+2,-1]

#for i in range(0,N):
#      xcord[i]=P[1][3*i,-1]
#      ycord[i]=P[1][3*i+1,-1]
#      zcord[i]=P[1][3*i+2,-1]

ax = plt.axes(projection='3d')
ax.set_zlabel(r'Z', fontsize=20)
ax.scatter3D(xcord, ycord, zcord)
ax.set_xlim(-2E-5,2E-5)
ax.set_ylim(-2E-5,2E-5)
ax.set_zlim(-2E-5,2E-5)

# Save Result
#np.save('P.npy',(P[0],P[1]))

A=np.array([P.t,P.y],dtype=object)
np.save('P.npy',A)