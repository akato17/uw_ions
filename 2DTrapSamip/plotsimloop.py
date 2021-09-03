# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:46:52 2021

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

while ionnum<N:
    #Want to see magnitude of Speed at each time, for one of the ions
    N=X['N']
    Y=X['Y']
    t=X['t']
    
    vx1= Y[3*N+3*ionnum]
    vy1= Y[3*N+1+3*ionnum]
    vz1= Y[3*N+2+3*ionnum]
    
    #ion's position
    x=Y[0+3*ionnum]
    y=Y[1+3*ionnum]
    z=Y[2+3*ionnum]
    
    
    
    #Find magnitude of distance, speed, Kinetic Energy, and Temperature
    magP= np.sqrt(x**2+y**2+z**2)
    
    magV= np.sqrt(vx1**2+vy1**2+vz1**2)
    
    sign=np.sign(y)
    theta= np.arccos(x/magP)*sign #theta goes from -pi to pi
    #Getting rid of the first half of the velocities
    # halfvec=np.arange(0,int(len(magV)/2))
    
    # magV=np.delete(magV, halfvec)
    # t=np.delete(t, halfvec)
    
    KE= (1/2)*m*magV**2
    T=m*magV**2/(3*1.38064852e-23)# denominator is boltzman-constant, this is only for 1 ion
    Td=hbar*laser.gamma/(2*1.38064852e-23)#This is doppler temperature
    
    # '''Trying to Get Time-Average Temp'''
    # #Basically cuts elements of 'T' down so it is divisible by 'bins', then averages points in groupings of 'binsize'
    # #AvgTt is a time scaled for the binsize
    # bins=500
    # binsize=int(len(T)/bins)
    # remainder=len(T)%bins
    # index=np.arange(len(T)-remainder+1,len(T)+1)
    # Tprime=np.delete(T,index-1)
    # Tprime=Tprime.reshape(bins,binsize)
    # AverageT=np.zeros(bins)
    # for i in range(0,bins):
    #     AverageT[i]=np.average(Tprime[i,:])
    # AvgTt=np.arange(0,len(AverageT))*tstep*binsize + twindow/2
    

    # #AvgTt is a time scaled for the binsize
    # bins=500
    # binsize=int(len(theta)/bins)
    # remainder=len(theta)%bins
    # index=np.arange(len(theta)-remainder+1,len(theta)+1)
    # thetaprime=np.delete(theta,index-1)
    # thetaprime=thetaprime.reshape(bins,binsize)
    # Averagetheta=np.zeros(bins)
    # for i in range(0,bins):
    #     Averagetheta[i]=np.average(thetaprime[i,:])
    # Avgthetat=np.arange(0,len(Averagetheta))*tstep*binsize
    
    '''Plotting magP and theta against t'''
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(t,magP)
    ax[0].set_title('R')

    ax[1].plot(t,theta)
    ax[1].set_title('Avg Theta')
    
    
    #total average T
    TotaverageT=np.average(T)
    #average distance
    Pavg= np.average(magP)
    
    print('Ion', ionnum)
    #print('Dist=',Pavg)

    #print('Total Average Temperature=', TotaverageT)
    
    
    ionnum=ionnum+1
