#!/usr/bin/env python
# coding: utf-8

# ## This document is to simulate only the geometry of the Elliptical Mirror Trap

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import time
from numba import njit,prange


# In[ ]:


@njit(boundscheck=True)
def trap(Vrf, Vdc):
    
    #Define the parameters that make up your trap
    eps = 0.97; #roughly the eccentricity
    f = 1; #flattening
    a = 0.05; #semi-minor axis
    c = 0.25*a; #distance from center to pole, semi major
    Rmin = 0.0125; #radius of the waist of the ellipsoid
    m = 0.01; #additional mirror thickness horizontal
    t0 = -0.04; #The same as b, but negitive
    b0 = 0.0509; #minus sign in BC
    t = -0.025; #additional mirror thickness vertical for the top
    b = 0.04; #additional mirror thickness vertical for the bottom
    b2 = 0.025; #needs to be the same as t, but should always be positive, even if t is negitive
    t2 =  0.025;
    Hneedle = -0.0009; #height of needle inside trap up until the beginning of the needle tip
    Rneedle = 0.00023; #radius of needle
    HneedleTip = 0.0024; # height of needle tip from needle body
    RneedleTip = 0.0001; # The ball on top of the needle
    Bneedle = b0; #where the needle begins

     
    
    #Properties of our grid
    gmax = 0.052 #physical size of the space simulated (meters)
    points = 401 # number of points that are being simulated
    steps = points - 1 # just the number of points minus one
    x = np.linspace(-gmax/2, gmax/2, points) #chosing how much space in the x, y, z direction, where it begins and where it ends
    y = np.linspace(-gmax/2, gmax/2, points)
    z = np.linspace(-gmax, 0, points)
    size = len(x) # telling it how to organize these spaces
    sizeZ = len(z)
    xmid = int((size)/2)
    ymid = int((size)/2)
    zmid = int((sizeZ)/2) 
    lattice_points=np.zeros((size**3,3))
    marker=0 #What do these mean?
    
    V0 = np.zeros((size, size, size)) #initialize the matrix for electric potential
    V0dc= np.zeros((size, size, size))
    V0_temp=np.zeros((size, size, size))
    V0dc_temp=np.zeros((size, size, size))
    cords=np.zeros((size, size, size, 3))
    iterations = 600 #number of iterations
    CenterV = np.zeros((iterations,1)) #keep track of the potential value at the center at each iteration
    
    # Now define the boundry conditions
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                cords[i, j, k] = [i, j, k]
                
                #The ellipsoidal mirror part 1
                if ((z[k] > (a/c)* np.sqrt(c**2 - x[i]**2 -y[j]**2)) and (z[k] < t) and (z[k] > -b)): #the top half of the ellipsoid
                    V0[i, j, k] = Vrf[0]
                    V0dc[i, j, k] = Vdc[0]
                elif ((z[k] < -(a/c)* np.sqrt(c**2 - x[i]**2 -y[j]**2)) and (z[k] < t) and (z[k] > -b)): #the bottom half of the ellipsoid
                    V0[i, j, k] = Vrf[0]
                    V0dc[i, j, k] = Vdc[0]
                elif ((y[j] > np.sqrt(Rmin**2 - x[i]**2)) and (y[j] < Rmin + m) and (z[k] < t) and (z[k] > -b)): #other part of mirror going around the top of ellipsoid
                    V0[i, j, k] = Vrf[0]
                    V0dc[i, j, k] = Vdc[0]
                elif ((y[j] < -np.sqrt(Rmin**2 - x[i]**2)) and (y[j] > -Rmin - m) and (z[k] < t) and (z[k] > -b)):#other part of mirror going around the bottom of ellipsoid
                    V0[i, j, k] = Vrf[0]
                    V0dc[i, j, k] = Vdc[0]
                elif ((x[i] < Rmin + m) and (x[i] > Rmin) and (y[j] < Rmin + m) and (y[j] > -Rmin - m) and (z[k] < t) and (z[k] > -b)): # to make an even square around ellipsoid
                    V0[i, j, k] = Vrf[0]
                    V0dc[i, j, k] = Vdc[0]
                elif ((x[i] < - Rmin) and (x[i] > -Rmin - m) and (y[j] < Rmin + m) and (y[j] > -Rmin - m) and (z[k] < t) and (z[k] > -b)): # to make an even square around ellipsoid
                    V0[i, j, k] = Vrf[0]
                    V0dc[i, j, k] = Vdc[0]
                    
                #The ellipsoidal mirror part 2
                elif ((z[k] > (a/c)* np.sqrt(c**2 - x[i]**2 -y[j]**2)) and (z[k] < t2) and (z[k] > -b2)): #the top half of the ellipsoid
                    V0[i, j, k] = Vrf[1]
                    V0dc[i, j, k] = Vdc[1]
                elif ((z[k] < -(a/c)* np.sqrt(c**2 - x[i]**2 -y[j]**2)) and (z[k] < t2) and (z[k] > -b2)): #the bottom half of the ellipsoid
                    V0[i, j, k] = Vrf[1]
                    V0dc[i, j, k] = Vdc[1]
                elif ((y[j] > np.sqrt(Rmin**2 - x[i]**2)) and (y[j] < Rmin + m) and (z[k] < t2) and (z[k] > -b2)): #other part of mirror going around the top of ellipsoid
                    V0[i, j, k] = Vrf[1]
                    V0dc[i, j, k] = Vdc[1]
                elif ((y[j] < -np.sqrt(Rmin**2 - x[i]**2)) and (y[j] > -Rmin - m) and (z[k] < t2) and (z[k] > -b2)):#other part of mirror going around the bottom of ellipsoid
                    V0[i, j, k] = Vrf[1]
                    V0dc[i, j, k] = Vdc[1]
                elif ((x[i] < Rmin + m) and (x[i] > Rmin) and (y[j] < Rmin + m) and (y[j] > -Rmin - m) and (z[k] < t2) and (z[k] > -b2)): # to make an even square around ellipsoid
                    V0[i, j, k] = Vrf[1]
                    V0dc[i, j, k] = Vdc[1]
                elif ((x[i] < - Rmin) and (x[i] > -Rmin - m) and (y[j] < Rmin + m) and (y[j] > -Rmin - m) and (z[k] < t2) and (z[k] > -b2)): # to make an even square around ellipsoid
                    V0[i, j, k] = Vrf[1]
                    V0dc[i, j, k] = Vdc[1]
                    
                    #The ellipsoidal mirror part 3
                elif ((z[k] > (a/c)* np.sqrt(c**2 - x[i]**2 -y[j]**2)) and (z[k] < t0+0.02) and (z[k] > -b0+0.02)): #the top half of the ellipsoid
                    V0[i, j, k] = Vrf[2]
                    V0dc[i, j, k] = Vdc[2]
                elif ((z[k] < -(a/c)* np.sqrt(c**2 - x[i]**2 -y[j]**2)) and (z[k] < t0) and (z[k] > -b0)): #the bottom half of the ellipsoid
                    V0[i, j, k] = Vrf[2]
                    V0dc[i, j, k] = Vdc[2]
                elif ((y[j] > np.sqrt(Rmin**2 - x[i]**2)) and (y[j] < Rmin + m) and (z[k] < t0) and (z[k] > -b0)): #other part of mirror going around the top of ellipsoid
                    V0[i, j, k] = Vrf[2]
                    V0dc[i, j, k] = Vdc[2]
                elif ((y[j] < -np.sqrt(Rmin**2 - x[i]**2)) and (y[j] > -Rmin - m) and (z[k] < t0) and (z[k] > -b0)):#other part of mirror going around the bottom of ellipsoid
                    V0[i, j, k] = Vrf[2]
                    V0dc[i, j, k] = Vdc[2]
                elif ((x[i] < Rmin + m) and (x[i] > Rmin) and (y[j] < Rmin + m) and (y[j] > -Rmin - m) and (z[k] < t0) and (z[k] > -b0)): # to make an even square around ellipsoid
                    V0[i, j, k] = Vrf[2]
                    V0dc[i, j, k] = Vdc[2]
                elif ((x[i] < - Rmin) and (x[i] > -Rmin - m) and (y[j] < Rmin + m) and (y[j] > -Rmin - m) and (z[k] < t0) and (z[k] > -b0)): # to make an even square around ellipsoid
                    V0[i, j, k] = Vrf[2]
                    V0dc[i, j, k] = Vdc[2]
                    
                #The needle
                elif ((y[j] < np.sqrt(Rneedle - x[i]**2)) and (x[i] < Rneedle) and (x[i] > -Rneedle) and (y[j] > -np.sqrt(Rneedle - x[i]**2)) and (y[j] < Rneedle) and (y[j] > -Rneedle) and (z[k] > -Bneedle) and (z[k] < -Bneedle + Hneedle)):
                    V0[i, j, k] = Vrf[3]
                    V0dc[i, j, k] = Vdc[3]
                #The needle tip
                elif ((z[k] < - np.sqrt(x[i]**2 + y[j]**2)*12.5) and (x[i] < Rneedle) and (x[i] > -Rneedle) and (y[j] > -Rneedle) and (y[j] < Rneedle) and (z[k] > -Bneedle + Hneedle) and (z[k] < -Bneedle + Hneedle + HneedleTip)):
                    V0[i, j, k] = Vrf[3]
                    V0dc[i, j, k] = Vdc[3]
                #the very tip of the needle (to avoid fringing effects)
                elif ((z[k] > -Bneedle + Hneedle + HneedleTip - 0.0002) and (z[k] < -Bneedle + Hneedle + HneedleTip + 0.0002) and (z[k] < np.sqrt(RneedleTip**2 - x[i]**2 - y[j]**2))):
                    V0[i, j, k] = Vrf[3]
                    V0dc[i, j, k] = Vdc[3]
                


                else:
                    V0[i, j, k] = 0
                    V0dc[i, j, k] = 0
                    lattice_points[marker]=[i,j,k]
                    marker+=1
                    
                    
    lattice_points=lattice_points[0:marker] # what does this mean?    
    
    V0_temp=V0    
    V0dc_temp=V0dc
    
    return V0,V0dc,CenterV,lattice_points,cords


# In[ ]:


#Voltages used in the simulation
RF=np.array([1, 1, 1, 1]) #corresponding to each portion of the mirror geometry
DC=np.array([1, 1, 1, 1])

start=time.time() # Records the time started.

A=trap(RF,DC) # Runs the simulation

print("The time elapsed is ",time.time()-start, "seconds")


# In[ ]:


#Plotting the resulting geometry

plt.imshow(A[0][:,200,:], cmap=plt.cm.get_cmap('viridis', 20)) #plotting the x-z axis cross section for the RF

