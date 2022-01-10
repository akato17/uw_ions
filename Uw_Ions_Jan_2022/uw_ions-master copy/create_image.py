# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:01:07 2020

@author: Alex K
This file exists to make images from trajectory data (like odeint output)
"""
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter.filedialog import askopenfilename
from numba import njit
# from numba import jit
####see website
from astropy.convolution import AiryDisk2DKernel, convolve
from PyQt5.QtWidgets import QFileDialog


'''Loading from file, pick simulation data file'''


filename=QFileDialog.getOpenFileName()[0]

X=np.load(filename,allow_pickle=True)


N=X['N']
Y=X['Y']
t=X['t']
#####extract number of ions
N=int(len(Y)/6)
length=len(Y[0])
#####extract position data
Npoints=int(len(Y[0]))
# L=int(2.5e6)###desired number of points to plot
L1=700000
L2=100000

#####number of pixels on ccd camera
n_pixels=256
#####physical size of image
image_area=256/1.78*1e-6
####funtion to only use a selected number of data points (L). Input entire dataset X
def trim_data(X,L1,l2):
      Npoints=int(len(X[0]))
      A=X[0:3*N]
      B=A.transpose()[Npoints-L1:Npoints-L2]
      return B.transpose()
#####function to convert data to pixel format input for pixel coords
def prepixeldata(X):
      Npoints=len(X[0])
      # data=X[1][0:3*N]
      # data=X[0:3*N]
      A=X.transpose().reshape((Npoints*N,3))#[int(Npoints-L):int(Npoints)]
      return A

"""
set up ccd image
"""
#pixel to um scale
scale=n_pixels/image_area
#####converts coordinates to pixels
def pixel_cords(X):
      #takes Nx3 numpy array and converts to NX2 Pixel array
      x=X[:,0]
      y=X[:,1]
      pixels=(np.stack((x,y)).transpose()/image_area*n_pixels+n_pixels/2)
      return pixels.astype(np.int32)

def pixel_cords_rot(X,xcenter=0,ycenter=0,angle=72.5*np.pi/180):
      #takes Nx3 numpy array and converts to NX2 Pixel array
      x=X[:,0]
      y=X[:,1]
      cords=np.stack((x,y)).transpose()
      cords_rot=rotate_set((0,0),cords,angle)
      cords_rot[:,1]*=-1
      pixels=(cords_rot/image_area*n_pixels+n_pixels/2)
      
      pixels[:,0]+=36
      pixels[:,1]+=20
      
      return pixels.astype(np.int32)
def spatial_cords(X):
      #takes Nx3 numpy array and converts to NX2 Pixel array
      x=X[:,0]
      y=X[:,1]
      pixels=(np.stack((x,y)).transpose())
      return pixels.astype(np.int32)
#####return image based on trajectory
@njit
def image_from_pixels(X):
      image=np.zeros((n_pixels,n_pixels))
      for i in X:
            if np.max(i)<=n_pixels:
                image[i[0],i[1]]+=1
      return image
# def image_microns(image,scale,xcenter,ycenter,xpixcent,ypixcent):
    
####blur and add airy rings to make more realistic
def convolve_image(X):
      airy=0.61*493e-9/.28 #radius of airy disk
      r=airy*scale
      disc=AiryDisk2DKernel(int(r))
      
      return  convolve(X,disc)

def trajectories_2d(X,L):
      Npoints=int(len(X[0]))
      A=X[1][0:3*N]
      B=A.transpose()[Npoints-L:Npoints].transpose()
     
      ####list of indices to delete
      index=[]
      for i in range(0,len(A)):            
            if i%3==2:
                  continue
            elif i%3==0 or i%3==1:
                  index.append(i)
      # index=np.array(index)
      # # print(index)
      
      
      return B[index].reshape(N,2,L)
@njit
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy
@njit
def rotate_set(origin, points, angle):
    points_rot=np.zeros((len(points),2))
    for i in range(0,len(points)):
        points_rot[i]=rotate(origin,points[i],angle)
    return points_rot
      
###example plot all ion trajectories in xy plane
# Q=trajectories_2d(X,100000)
# for i in range(0,len(Q)):
#     plt.plot(Q[i][0],Q[i][1])           
      
      
######choose colormap
plt.set_cmap('Blues_r')
#########convert actual trajectory data to an image
C=trim_data(Y,L1,L2)
A=prepixeldata(C)
B=pixel_cords(A)
D=pixel_cords_rot(A,xcenter=0,ycenter=0,angle=(72.5)*np.pi/180)
pic=image_from_pixels(D)
#####plot the simulated image
plt.imshow(convolve_image(pic),origin='lower')
# plt.clim(0,25000)
# plt.imshow(pic)

