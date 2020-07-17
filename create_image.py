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
data=X[1][0:3*N]
####start data point
start=800000
###number of data points to include in image
duration=20000
#####data for plot
A=data.transpose().reshape((length*N,3))[start:start+duration]


#####number of pixels on ccd camera
n_pixels=512
#####physical size of image
image_area=120e-6



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
######choose colormap
plt.set_cmap('Blues_r')
#########convert actual trajectory data to an image
B=pixel_cords(A)
pic=image_from_pixels(B)
#####plot the simulated image
plt.imshow(convolve_image(pic))

# plt.imshow(pic)

