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
path=''
###########################load your file
root=tkinter.Tk()
filename = askopenfilename()
root.destroy()

X=np.load(filename,allow_pickle=True)


#####extract number of ions
N=int(len(X[1])/6)
length=len(X[0])
#####extract position data

# L=int(2.5e6)###desired number of points to plot
L=50000

#####number of pixels on ccd camera
n_pixels=512
#####physical size of image
image_area=120e-6
####funtion to only use a selected number of data points (L). Input entire dataset X
def trim_data(X,L):
      Npoints=int(len(X[0]))
      A=X[1][0:3*N]
      B=A.transpose()[Npoints-L:Npoints]
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
#####return image based on trajectory
@njit
def image_from_pixels(X):
      image=np.zeros((n_pixels,n_pixels))
      for i in X:
           
            image[i[0],i[1]]+=1
      return image
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
      
###example plot all ion trajectories in xy plane
# Q=trajectories_2d(X,100000)
# for i in range(0,len(Q)):
#     plt.plot(Q[i][0],Q[i][1])           
      
      
######choose colormap
plt.set_cmap('Blues_r')
#########convert actual trajectory data to an image
C=trim_data(X,L)
A=prepixeldata(C)
B=pixel_cords(A)
pic=image_from_pixels(B)
#####plot the simulated image
plt.imshow(convolve_image(pic))
# plt.clim(0,25000)
# plt.imshow(pic)

