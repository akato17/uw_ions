# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 14:36:01 2021

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

'''This File Calculates T based on the definition of T in the Marciante Paper
This will get rid of micromotion.'''

'''Loading from file, pick simulation data file'''

root=tkinter.Tk()
filename = askopenfilename()
root.destroy()

X=np.load(filename,allow_pickle=True)


N=X['N']
Y=X['Y']
t=X['t']


'''Resizeing data to be in units of rf period'''
rfsize=int(period/tstep) #number of indexes for 1 rf period
excess= len(Y[0])%rfsize
delarr= np.arange(len(Y[0])-excess,len(Y[0]))

Y=np.delete(Y,delarr,1)#Y is now divisible by rfsize

'''Average each RF period'''
TransY=np.zeros((int(len(Y)), int(len(Y[0])/rfsize)))
for i in range(0,len(Y)):
    Row=Y[i]
    Rowrs=Row.reshape(int(len(Row)/rfsize),rfsize)
    avgRow=np.zeros(int(len(Row)/rfsize))
    for j in range(0,int(len(Row)/rfsize)):
        avgRow[j]=np.average(Rowrs[j,:])
    TransY[i]=avgRow


RfVel=np.delete(TransY,np.arange(0,3*N),0)#getting rid of positions

'''FirstTerm'''
RfVelsq=RfVel**2

Firstx=np.zeros(len(RfVel[0]))
for i in range(0,len(RfVel[0])):
    Sum=0
    for j in range(0,N):
        Sum+= RfVelsq[3*j,i]
    Firstx[i]=Sum/N
    
Firsty=np.zeros(len(RfVel[0]))
for i in range(0,len(RfVel[0])):
    Sum=0
    for j in range(0,N):
        Sum+= RfVelsq[3*j+1,i]
    Firsty[i]=Sum/N
    
Firstz=np.zeros(len(RfVel[0]))
for i in range(0,len(RfVel[0])):
    Sum=0
    for j in range(0,N):
        Sum+= RfVelsq[3*j+2,i]
    Firstz[i]=Sum/N
    
First=np.array([Firstx,Firsty,Firstz])

'''SecondTerm'''

Secondx=np.zeros(len(RfVel[0]))
for i in range(0,len(RfVel[0])):
    Sum=0
    for j in range(0,N):
        Sum+= RfVel[3*j,i]
    Secondx[i]=Sum/N
    
Secondy=np.zeros(len(RfVel[0]))
for i in range(0,len(RfVel[0])):
    Sum=0
    for j in range(0,N):
        Sum+= RfVel[3*j+1,i]
    Secondy[i]=Sum/N
    
Secondz=np.zeros(len(RfVel[0]))
for i in range(0,len(RfVel[0])):
    Sum=0
    for j in range(0,N):
        Sum+= RfVel[3*j+2,i]
    Secondz[i]=Sum/N
    
Second=np.array([Secondx,Secondy,Secondz])**2


T=(m/1.38064852e-23)*(First-Second)

print('Avg Tx=', np.average(T[0]))
print('Avg Ty=', np.average(T[1]))
print('Avg Tz=', np.average(T[2]))