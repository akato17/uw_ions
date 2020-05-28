# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:01:07 2020

@author: barium
This file exists to make images from trajectory data (like odeint output)
"""
import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
####see website
from astropy.convolution import AiryDisk2DKernel, convolve
path=''
# X=np.load('change_name.npy',allow_pickle=True)

# N=int(len(X.transpose())/6)
# data=X.transpose()[0:3*N]
# length=len(data[0])
# start=890000
# duration=990000
# A=data.transpose().reshape((length*N,3))[start:start+duration]



#####for using solve_ivp insetad
X=np.load('change_name.npy',allow_pickle=True)
duration=3000000

N=int(len(X[1])/6)
length=len(X[0])
start=200000
duration=50000
data=X[1][0:3*N].transpose()[start:start+duration]
# start=length-duration


A=data.reshape((duration*N,3))



n_pixels=512

image_area=120e-6



"""
set up ccd image
"""
scale=n_pixels/image_area
def pixel_cords(X):
      #takes Nx3 numpy array and converts to NX2 Pixel array
      x=X[:,0]
      y=X[:,1]
      pixels=(np.stack((x,y)).transpose()/image_area*n_pixels+n_pixels/2)
      return pixels.astype(np.int32)
# @jit jit enable this later
def image_from_pixels(X):
      image=np.zeros((n_pixels,n_pixels))
      for i in X:
           
            image[i[0],i[1]]+=1
      return image
def convolve_image(X):
      airy=0.61*493e-9/.28 #radius of airy disk
      r=airy*scale
      disc=AiryDisk2DKernel(int(r))
      
      return  convolve(X,disc)

plt.set_cmap('Blues_r')

B=pixel_cords(A)
pic=image_from_pixels(B)

plt.imshow(convolve_image(pic))

# plt.imshow(pic)

