# -*- coding: utf-8 -*-
"""
@author: Alex K, Zeyu Y
Create a series of images which reflects dynamics of ions.
"""
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter.filedialog import askopenfilename
from numba import njit
import laser
from userconfig import *
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

L=int(10e-9/tstep) # Length of each segments (denominator is the tstep)
end_pointer = int(len(X[0])) # Pointer to the end of the current segment

#####number of pixels on ccd camera
n_pixels=512
#####physical size of image
image_area=12e-5
####funtion to only use a selected number of data points (L). Input entire dataset X
def trim_data(X,L,end_pointer):
      A=X[1][0:3*N]
      C=X[1][3*N:6*N]
      B=A.transpose()[end_pointer-L:end_pointer]
      D=C.transpose()[end_pointer-L:end_pointer]
      end_pointer -= L
      return B.transpose(),D.transpose(),end_pointer
#####function to convert data to pixel format input for pixel coords
def prepixeldata(X,V):
      Npoints=len(X[0])
      # data=X[1][0:3*N]
      # data=X[0:3*N]
      A=X.transpose().reshape((Npoints*N,3))#[int(Npoints-L):int(Npoints)]
      B=V.transpose().reshape((Npoints*N,3))
      return A,B

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
#@njit
def image_from_pixels(X,V):
      image=np.zeros((n_pixels,n_pixels))
      for i in range(0,len(X)):
            weight = laser.laser_vect(V[i,:],1)
            image[X[i,0],X[i,1]]+=int(np.sqrt(weight[0]**2+weight[1]**2+weight[2]**2)*10**20)
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
      
# Repeat for desired number of segments
for i in range(0,40):
      # Get the current segement
      B,D,end_pointer=trim_data(X,L,end_pointer) 
      ######choose colormap
      plt.set_cmap('Blues_r')
      #########convert actual trajectory data to an image
      Xq,V=prepixeldata(B,D)
      A=pixel_cords(Xq)
      pic=image_from_pixels(A,V)
      #####plot the simulated image
      plt.imshow(convolve_image(pic))
      plt.savefig('z'+str(40-i)+'.png')
      plt.close('all')
      # plt.clim(0,25000)
      # plt.imshow(pic)

