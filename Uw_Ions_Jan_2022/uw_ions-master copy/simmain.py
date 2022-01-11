# -*- coding: utf-8 -*-
"""
@author: Samip
@desc An example of using the functions.
@date 9/3/2021
"""

import numpy as np
import fieldfunc as ff
import simfunc as sf
import matplotlib.pyplot as plt
import scipy.integrate as INT
from Field import Field
from userconfig import *
import laser
import laser2
# import tkinter
from PyQt5.QtWidgets import QFileDialog

# import the field, and change the mode to create a new one
# real_field = Field(mode = 'create',paramsRF_file = 'paramsRF.npy', paramsDC_file = 'paramsDC.npy')
#real_field.save_params()

# Run simulation
N = 54
#Initialize condition
IC = sf.initialize_ions(N)

#  # OR start with another file
# load = np.load('54_ions.npz',allow_pickle=True)#[1]
# IC = np.zeros(6*N)
accf=np.zeros(3*N)
tstart=0
twindow=5e-3
# # take the final states

# IC=np.load(filename,allow_pickle=True)
# for i in range(0,len(load)):
#       IC[i] = load[i][-1]
# filename = askopenfilename()
# IC=np.load(filename,allow_pickle=True)


# "loading from file"
# print('Pick ICs')
# # root=tkinter.Tk()
# filename=QFileDialog.getOpenFileName()[0]
# # # # # # root.destroy()

# ##############################
# print('loading IC')
# load=np.load(filename,allow_pickle=True)
# print('assigning IC')
# IC=load['Y'][:,-1]
# # for i in range(0,len(load['Y'])):
# #       IC[:] = load['Y'][][-1]
# # print(IC)
# tstart=load['t'][-1]

# accf=np.array(load['accf'])
# ####################################
# #P = INT.solve_ivp(lambda t, y: sf.Newton4(t,y,N, real_field.paramsDC, real_field.paramsRF), [0,twindow], y0=IC, t_eval=np.arange(0, twindow, tstep), method='RK45',max_step=tstep)
# res = np.zeros(200)
# fake=np.array([0,0,0,0,0,0,0,0])


# Freq vect for LIR  (Laser Induced Radiation)
#fm = np.linspace(1*10**5,4*10**5,200)

# Newton 3
#P = INT.solve_ivp(lambda t, y: sf.Newton3(t,y,N, real_field.paramsDC, real_field.paramsRF,0), [0,twindow], y0=IC, t_eval=np.arange(0, twindow, tstep), method='RK45',max_step=tstep)
Side_V=Side_V
Endcap_V=Endcap_V
RFamp=RFamp
print('starting sim')
# Leap Frog

P= sf.leap_frog3(N, IC, Side_V,Endcap_V,RFamp, tstart,twindow,accf)



#Code used for LIR and save images
# for i in range(0,200):
#       #P = sf.leap_frog(N, IC, real_field.paramsDC *3/10, real_field.paramsRF, fm[i])
#       P = INT.solve_ivp(lambda t, y: sf.Newton3(t,y,N, real_field.paramsDC, real_field.paramsRF,fm[i]), [0,twindow], y0=IC, t_eval=np.arange(0, twindow, tstep), method='RK45',max_step=tstep)
#       vel = P[1][3 * N : 6 * N, -1]
#       for j in range(0,N):
#             res[i] += np.sqrt(vel[3*j]**2+vel[3*j+1]**2+vel[3*j+2]**2)
#       if(i%10 == 0):
#             xcord=np.zeros(N,dtype=np.float64)
#             ycord=np.zeros(N,dtype=np.float64)
#             zcord=np.zeros(N,dtype=np.float64)
#             for j in range(0,N):
#                   xcord[j]=P[1][3*j,-1]
#                   ycord[j]=P[1][3*j+1,-1]
#                   zcord[j]=P[1][3*j+2,-1]
#             ax = plt.axes(projection='3d')
#             ax.set_zlabel(r'Z', fontsize=20)
#             ax.set_xlim3d(-30e-6, 30e-6)
#             ax.set_ylim3d(-30e-6,30e-6)
#             ax.set_zlim3d(-30e-6,30e-6)
#             ax.scatter3D(xcord, ycord, zcord)
#             plt.savefig(str(i)+'.png')
#             plt.close('all')

Y=P[1]
t=P[0]
accf=P[2]
'''Truncating Data to n sigfigs, will not save space'''
#Y=sf.truncate_sigfigs(Y,4)#n=4 right now
# t=sf.truncate_sigfigs(t,4)

'''Changes data to float 32, less data'''
Y=np.array(Y,dtype=np.float32) 
t=np.array(t,dtype=np.float32)
accf=np.array(accf,dtype=np.float32)
'''Saving Options'''
# Saving P once I run this 
# X=np.array([P.t,P.y])
# np.save("test.npy",X)

'''NewtonSim'''
# X=P.y[:,-1]#saving last collumn of P.y, to get new IC's
# np.save("ICrecent.npy",X)

# #Saving to show discrete laser with already cool ions
#np.savez('Run_recent', t=P.t, Y=P.y, N=N)

#saving IC from leapfrog2, i always use this
X=P[1]
X=X[:,-1]
np.save("ICrecent.npy",X)
'''Leap Frog'''

np.savez('Run_recent', t=t, Y=Y, N=N,accf=accf)


print('Sim Finished, Run_recent and ICrecent saved, open them in different code')

'''Open plotsim to simulate latest sim'''
