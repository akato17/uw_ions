# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:29:28 2020

@author: barium
"""
import numpy as np
import MDfunc as md
from MDfunc import Newton3,FHYP_vect,Coulomb_vect,Coulomb_jit,laser_vect,F_DC,Newton4
from MDfunc import Newton5
from scipy.constants import epsilon_0
from scipy.constants import k
from scipy.constants import c
from scipy.constants import hbar
from scipy.constants import h
from scipy.constants import physical_constants
import matplotlib.pyplot as plt
from numba import njit, prange
import scipy.integrate as INT

import time
##number of ions
N=28

path='change_name.npy'
M=physical_constants['atomic mass constant'][0]
mb=138*M



'''
Laser cooling parameters for constant force (not for sweeping)

'''

###laser wavelength
wav=493e-9
#laser wave number
k_number=2*np.pi/wav
###wave vector
k_vect=1/np.sqrt(3)*np.array((1,1,1))
K=k_vect*k_number
###laser frequency
freq=c/wav
##delta i.e. detuning
####excited state lifetime for p1/2 state
tau=10e-9 ####I need an actual referenece for this
####gamma
gamma=1/tau
###saturation intensity
Isat=np.pi*h*c/(3*wav**2*tau)
####laser intensity
I=2*Isat
# I=0
delta=-0.5*gamma

###saturation parameter
# s0=I/Isat



"""
Trap parameters
"""
####secular frequencies if using no micromotion
# wx=.205e6*2*np.pi
# wy=.22e6*2*np.pi
# wz=.6e6*2*np.pi
#####hyperbolic trap parameters
r0=.002
z0=.0005
V=2000
####rf drive frequency
omega=10e6*2*np.pi 

"""
Magnetic field
"""
B=1e-3*np.array([0,0,1])
"""
initial conditions parameters
"""
### initial conditions of barium ions, temperature, boltzman mean and std
Tb=500
# mu_barium=np.sqrt(8*k*Tb/(np.pi*mb))
# sigma_barium=np.sqrt(k*Tb/(2*mb))
###temperature of virtual gas

###size of grid where initial positions may start
start_area=100e-6

"""
if using collisions method of damping
"""
####mass of cold gas for ccd reproduction images
mg=mb/100
T=2e-3
###mean and std of boltzmann distribution for virtual gas
mu=np.sqrt(8*k*T/(np.pi*mg))
sigma=np.sqrt(k*T/(2*mg))
######integration step
t_int=5e-9
###########################if not loading from a file
####total time
Tfinal=.0075
####timestep at which to record data (cant be lower than t_int and should be a multiple of it)
t_step=10e-9
####time variable to start at (so you don't record the whole cooling part if you don;t want to)
t_start=.005
#####times at which to record data
trange=[0,Tfinal]
t2=np.arange(t_start, Tfinal, t_int)
# IC=md.initialize_ions(N,start_area,Tb)
IC=md.initialize_ions(N,start_area,Tb)
IC2=md.initialize_ions_noz(N,start_area,Tb)

in_path="28 ions periodic.npy"
data_load=np.load(in_path,allow_pickle=True)
eq=np.array(data_load[1][:,-1])
tSTART=data_load[0][-1]
trange3=[tSTART,tSTART+Tfinal]
t3=np.arange(tSTART+.14, Tfinal+tSTART, 2*t_step)
# the start time is the last time interval of the prvious simulation
#free up some RAM if you had input a large file
del data_load

start=time.time()
print("start")

# 
P = INT.solve_ivp(lambda t, y: Newton5(t,y,N), trange, y0=IC2, t_eval=t2, method='RK45',max_step=t_int)
print(time.time()-start)
#####if using solve_ivp
xcord=np.zeros(N)
ycord=np.zeros(N)
zcord=np.zeros(N)
for i in range(0,N):
      xcord[i]=P.y[3*i,-1]
      ycord[i]=P.y[3*i+1,-1]
      zcord[i]=P.y[3*i+2,-1]
ax = plt.axes(projection='3d')
ax.set_zlabel(r'Z', fontsize=20)
ax.set_xlim3d(-30e-6, 30e-6)
ax.set_ylim3d(-30e-6,30e-6)
ax.set_zlim3d(-30e-6,30e-6)
ax.scatter3D(xcord, ycord, zcord)
np.save(path,(P.t,P.y))
# from scipy.spatial import Voronoi, voronoi_plot_2d
# points=np.array((xcord,ycord)).transpose()
# vor = Voronoi(points)