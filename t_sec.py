# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:51:36 2020

@author: barium
post processing
"""
import numpy as np
from scipy.constants import k
from scipy.constants import physical_constants
import tkinter
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt



######choose a file
root=tkinter.Tk()
filename = askopenfilename()
root.destroy()
#####mass barium
M=physical_constants['atomic mass constant'][0]
mb=138*M
#####extract data
X=np.load(filename,allow_pickle=True)
#####number of ions
N=int(len(X[1])/6)
######total number of points
n_points=len(X[0])
####rf frequency (make sure you know this)
omega=10e6*2*np.pi
######period of micromotion
period=2*np.pi/(omega)
###
######time space between datapoints
t=10e-9
####
# points_per_interval=int((period/t)*100)
# number_of_intervals=int(n_points/points_per_interval)
# leftover=n_points%points_per_interval
# vel2=((X[1][:])[leftover:])**2
# data=(X[1][3*N:6*N]).transpose()[leftover:].transpose()
# # sec_temp

######find the secular temp accoring to Zhang2007
def sec_temp(X):
      ###number of period to average over
      nav=11
      #####number of points per averaging interval (period)
      points_per_interval=int((period/t)*nav)
      #####number of intervals to average over
      number_of_intervals=int(n_points/points_per_interval)
      ####exclude some points so we don't have any stragglers
      leftover=n_points%points_per_interval
      N=int(len(X[1])/6)
      ######data in a format we can average (excluding the leftover points)
      data=(X[1][3*N:6*N]).transpose()[leftover:].transpose()
      # data=data[3*N:6*N]
      #####reshape. axis 0 is for each ion, axis 1 is xyz dimensions, and axis 2 is the data organized by micromotion periods
      vels=data.reshape((N,3,number_of_intervals,points_per_interval))
      
      ####return the secular temperature of the crystal
      return mb/(3*N*k)*np.sum(np.sum(np.average(vels,axis=3)**2,axis=1),axis=0)

plt.plot(sec_temp(X))

