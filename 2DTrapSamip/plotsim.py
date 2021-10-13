# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:31:24 2021

@author: samip
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
import tkinter
from tkinter.filedialog import askopenfilename


'''Loading from file, pick simulation data file'''

root=tkinter.Tk()
filename = askopenfilename()
root.destroy()

X=np.load(filename,allow_pickle=True)


N=X['N']
Y=X['Y']
t=X['t']


ionnum=0 #which ion of the N you want to see the plots of. Starts at ion 0,1,2,3...
'''Plotting Options'''


# 3D plotting of last positions
# !!! if leap_frog, change P.t -> P[0] and P.y -> P[1]
xcord=np.zeros(N,dtype=np.float64)
ycord=np.zeros(N,dtype=np.float64)
zcord=np.zeros(N,dtype=np.float64)
for j in range(0,N):
      xcord[j]=Y[3*j,-1]
      ycord[j]=Y[3*j+1,-1]
      zcord[j]=Y[3*j+2,-1]
      
'''3d'''    
# ax = plt.axes(projection='3d')
# ax.set_zlabel(r'Z', fontsize=20)
# ax.set_xlim3d(-50e-6, 50e-6)
# ax.set_ylim3d(-50e-6,50e-6)
# ax.set_zlim3d(-50e-6,50e-6)
# ax.scatter3D(xcord, ycord, zcord)


'''2d'''
plt.scatter(xcord, ycord)
plt.show()
#Want to see magnitude of Speed at each time, for one of the ions

vx1= Y[3*N+3*ionnum]
vy1= Y[3*N+1+3*ionnum]
vz1= Y[3*N+2+3*ionnum]

#ion's position
x=Y[0+3*ionnum]
y=Y[1+3*ionnum]
z=Y[2+3*ionnum]

#Find magnitude of distance, speed, Kinetic Energy, and Temperature
#magP= np.sqrt(x**2+y**2+z**2)

#sign=np.sign(y)
#theta= np.arccos(x/magP)*sign #theta goes from -pi to pi


#magV= np.sqrt(vx1**2+vy1**2+vz1**2)

#Getting rid of the first half of the velocities
# halfvec=np.arange(0,int(len(magV)/2))

# magV=np.delete(magV, halfvec)
# t=np.delete(t, halfvec)

#KE= (1/2)*m*magV**2
#T=m*magV**2/(3*1.38064852e-23)# denominator is boltzman-constant, this is only for 1 ion
#Td=hbar*laser.gamma/(2*1.38064852e-23)#This is doppler temperature

'''Trying to Get Time-Average Temp'''
#Basically cuts elements of 'T' down so it is divisible by 'bins', then averages points in groupings of 'binsize'
#AvgTt is a time scaled for the binsize
# bins=500
# binsize=int(len(T)/bins)
# remainder=len(T)%bins
# index=np.arange(len(T)-remainder+1,len(T)+1)
# Tprime=np.delete(T,index-1)
# Tprime=Tprime.reshape(bins,binsize)
# AverageT=np.zeros(bins)
# for i in range(0,bins):
#     AverageT[i]=np.average(Tprime[i,:])
# AvgTt=np.arange(0,len(AverageT))*tstep*binsize

'''Trying to get Time-Average Theta'''
# Basically cuts elements of 'T' down so it is divisible by 'bins', then averages points in groupings of 'binsize'
# AvgTt is a time scaled for the binsize
# bins=1000
# binsize=int(len(theta)/bins)
# remainder=len(theta)%bins
# index=np.arange(len(theta)-remainder+1,len(theta)+1)
# Tprime=np.delete(theta,index-1)
# Tprime=Tprime.reshape(bins,binsize)
# AverageTh=np.zeros(bins)
# for i in range(0,bins):
#     AverageTh[i]=np.average(Tprime[i,:])
# AvgTht=np.arange(0,len(AverageTh))*tstep*binsize

# #total average T
# TotaverageT=np.average(T)
# #average distance
# Pavg= np.average(magP)

#Plotting distance with time, and also magnitude of velocity with time

'''Choose Plots'''
#fig, ax = plt.subplots(1,1,sharex=True)
# ax[0,0].plot(t,magP)
# ax[0,0].set_title('Distance')

# ax[0].plot(t,magV)
# ax[0].set_title('Speed')

# ax[1,0].plot(t,KE)
# ax[1,0].set_title('Kinetic Energy')

# ax[0].plot(t,T)
# ax[0].set_title('Instant Temp')

# ax.plot(AvgTt,AverageT)

# Tdplot=np.array([1,1])*Td
# Tdplottime=np.array([0,twindow])
# Tdplottime=np.array([twindow/2,twindow])
# ax.plot(Tdplottime,Tdplot)
# ax.set_title('Avg Temp (Doppler Temp)')



#%matplotlib qt #interactive plot

#plt.plot(t,theta)
# plt.plot(AvgTht,AverageTh)
# plt.xlabel('time')
# plt.ylabel('Av Theta')
# # #plt.set_title('Ion',ionnum)
# plt.show()

# print('Ion', ionnum)
# print('Dist=',Pavg)
#print('Doppler Temperature=',Td )
# print('Total Average Temperature=', TotaverageT)

# plt.plot(AvgTt,AverageT)
# plt.xlabel('time')
# plt.ylabel('Av Temp')
# #plt.set_title('Ion',ionnum)
# plt.show()

#print('Final Avg Temp=',AverageT[-1])