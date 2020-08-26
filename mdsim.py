# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:29:28 2020

@author: Alex K.
"""
import numpy as np
import MDfunc as md
from MDfunc import Newton3,Newton5,leap_frog
from MDfunc import omega as omg
import mdconst as mc
import matplotlib.pyplot as plt
from numba import njit, prange
import scipy.integrate as INT
from tkinter.filedialog import askopenfilename
import tkinter
import time

###load file. If True you will get prompted to open one. if False, it will use random initial conditions
load_file=True
#######set file name for saving after
path='change_name.npy'
##number of ions
N=6
'''
Integration Parameters
'''
######integration step
omega1=omg
# t_int=5e-9
#####set integration time to 1/20 of period
t_int=1/(omg/2/np.pi)/20
###########################if not loading from a file
####total time
Tfinal=.005
####timestep at which to record data (cant be lower than t_int and should be a multiple of it)
t_step=2*t_int
####time variable to start at (so you don't record the whole cooling part if you don;t want to)
t_start=0
#####times at which to integrate
trange=[0,Tfinal]
# Times to record data

t2=np.arange(t_start, Tfinal, t_int)

####mass of cold gas for ccd reproduction images
mg=mc.mb/100
T=2e-3
###mean and std of boltzmann distribution for virtual gas
mu=np.sqrt(8*mc.k*T/(np.pi*mg))
sigma=np.sqrt(mc.k*T/(2*mg))

"""
initial conditions parameters
"""
### initial conditions of barium ions, temperature, boltzman mean and std
Tb=150

###size of grid where initial positions may start
start_area=200e-6

###random initial conditions
IC=md.initialize_ions(N,start_area,Tb)
####random initial conditions if you want to start with all ions at Z=0
IC2=md.initialize_ions_noz(N,start_area,Tb)

#########if you are laoding a file
if load_file==True:
      root=tkinter.Tk()
      filename = askopenfilename()
      root.destroy()
      # in_path="28 ions periodic.npy"
      data_load=np.load(filename,allow_pickle=True)
      #extract number of ions
      N=int(len(data_load[1])/6)
      #find last position of all ions
      eq=np.array(data_load[1][:,-1])
      #set initial conditions to the last known position
      IC=eq
      ###set start time to be the last time of the previous simulation
      tSTART=data_load[0][-1]
      #new integration range beginning from previous start time
      trange=[tSTART,tSTART+Tfinal]
      ####new time to record data
      t2=np.arange(tSTART, Tfinal+tSTART, t_int*2)
      #free up some RAM if you had input a large file
      del data_load




######time your simulation
start=time.time()
print("simulation has started")
xcord=np.zeros(N,dtype=np.float64)
ycord=np.zeros(N,dtype=np.float64)
zcord=np.zeros(N,dtype=np.float64)

# ######if using RK45 or other integrator
# Q = INT.solve_ivp(lambda t, y: Newton3(t,y,N), trange, y0=IC, t_eval=t2, method='RK45',max_step=t_int)
# # P=np.zeros((3))
# A=np.array([Q.t,Q.y],dtype=object)
# np.save(path,A)

# for i in range(0,N):
#       xcord[i]=Q.y[3*i,-1]
#       ycord[i]=Q.y[3*i+1,-1]
#       zcord[i]=Q.y[3*i+2,-1]
#####if using leapfrog algorithm
P=leap_frog(N,Tfinal,t_int,np.array(IC))
np.save(path,(P[0],P[1]))
###find final coordinates and plot to check if there is a crystal (with solve_ivp)

for i in range(0,N):
      xcord[i]=P[1][3*i,-1]
      ycord[i]=P[1][3*i+1,-1]
      zcord[i]=P[1][3*i+2,-1]



#####how much time it took
print("simulation has finished and took",time.time()-start,"s")


####find final coordinates and plot to check if there is a crystal (with solve_ivp)


ax = plt.axes(projection='3d')
ax.set_zlabel(r'Z', fontsize=20)
ax.set_xlim3d(-30e-6, 30e-6)
ax.set_ylim3d(-30e-6,30e-6)
ax.set_zlim3d(-30e-6,30e-6)
ax.scatter3D(xcord, ycord, zcord)



