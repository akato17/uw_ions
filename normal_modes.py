# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:18:16 2020

@author: barium
this module is written to calculate the normal modes of oscillation. It assumes youalready 
have the equilibrium trajectories
"""
import numpy as np
from tkinter.filedialog import askopenfilename
import tkinter
from numba import njit
from matplotlib import pyplot as plt
from scipy.constants import e,physical_constants,epsilon_0
# 
#####open file dialog. Choose a trajectory data file or this wont work
####alternatively, manually type in filepath to np.load command
root=tkinter.Tk()
filename = askopenfilename()
root.destroy()
# load the file
X=np.load(filename,allow_pickle=True)
#####decipher how many ions were in the simulation
N=int(len(X[1])/6)
########mass of barium
M=physical_constants['atomic mass constant'][0]
mb=138*M
#####rf drive frequency
omega=2*np.pi*10e6
####length scale to make data dimensionless
L=((e**2)/(4*np.pi*epsilon_0*mb*omega**2))**(1/3)

###R must be dimensionless
Rt=np.transpose(X[1][0:3*N])[0]/(L)
###pick out the eq positions over one period of micromotion (omega)
R=np.transpose(X[1][0:3*N])[0:19]/L 
###pick out the eq positions over one period of micromotion
traj=X[1][17][0:19]
########input all matthieu paramaters. soon to be automatic from trajectory data
r_0=2e-3
z_0=.5e-3
q_z=-8*e*2000/(mb*np.sqrt(r_0**2+2*z_0**2)*omega**2)
q_x=-q_z/2
q_y=-q_z/2
Vend=20
r0=2e-3
z0=.5e-3
a_z=-16*e*Vend/((mb*np.sqrt(r0**2+2*z0**2))*omega**2)
a_x=-a_z/2-.1*a_z
a_y=-a_z/2+.1*a_z

a=np.array([a_z,a_y,a_z])
q=np.array([q_x,q_y,q_z])
#####average the ion's position over one micromotion period
Rav=np.average(R,axis=0)
##########sum functions just to make the K matrix from the paper
@njit
def sum1(R,i,j):
      return (R[3*i]-R[3*j])**2+(R[3*i+1]-R[3*j+1])**2+(R[3*i+2]-R[3*j+2])**2
@njit
def sum2(R,i,sigma,tau):
      A=0
      for k in range(0,int(len(R)/3)):
                  
            
                  
                  if k==i:
                        continue
                  else:
                        
                        A+=(R[3*i+sigma]-R[3*k+sigma])*(R[3*i+tau]-R[3*k+tau])/sum1(R,i,k)**(5/2)
      return A
@njit(boundscheck=True)
def sum3(R,i,sigma):
      A=0
      for k in range(0,int(len(R)/3)):
            
            if k==i:
                  continue
            else:
                  A+=(sum1(R,i,k)-3*(R[3*i+sigma]-R[3*k+sigma])**2)/sum1(R,i,k)**(5/2)
      return A
#########compute the K matrix
@njit     
def K(R):
      '''
      
takes an input vector of 3N ion positions and spits out the K matrix for those positions
      returns a 3Nx3N matrix
      

      '''
      N=int(len(R)/3)
      K=np.zeros((3*N,3*N))
      for k in range(0,len(R)):
            for l in range(0,len(R)):
                  sigma=k%3
                  tau=l%3
                  i=int(k/3)
                  j=int(l/3)
                  # print(i,j,sigma,tau)

                  if i!=j and sigma!=tau:
                        K[k,l]=-3*(R[3*i+sigma]-R[3*j+sigma])*(R[3*i+tau]-R[3*j+tau])/\
                              sum1(R,i,j)**(5/2)
                  elif i!=j and sigma==tau:
                        K[k,l]=(sum1(R,i,j)-3*(R[3*i+sigma]-R[3*j+sigma])**2)/\
                              sum1(R,i,j)**(5/2)
                  elif i==j and sigma!=tau:
                        K[k,l]=3*sum2(R,i,sigma,tau)
                  elif i==j and sigma==tau:
                        K[k,l]=-1*sum3(R,i,sigma)
      return K
#####compute the K matrix for each point over the period of micromotion
@njit
def Kt(R):
      Q=np.zeros((len(R),3*N,3*N))
      for i in range(0,len(R)):
            Q[i]=K(R[i])
            
      return Q
#######compiute the first two fourier components of the K matrix over one period
def F_K(K):
      """
      computes the fourier decomposition of K(t)
      assuming one period of data. must change if more
      """
      A=np.fft.fftn(K,axes=[0])
      a0=np.real(A[0])
      a1=np.real(A[1]+A[-1])
      a2=np.real(A[2]+A[-2])
      return a0,a1,a2
#######compute A and Q
@njit
def AQ(K_0,K_2):
      
      A=K_0
      Q=K_2
      for k in range(0,len(Q)):
            for l in range (0,len(Q)):
                  sigma=k%3
                  tau=l%3
                  i=int(k/3)
                  j=int(l/3)
                  if i!=j: continue
                  elif i==j:
                        if sigma==tau:
                              Q+=q[sigma]
                              A+=a[sigma]
                        else: continue
      return A,Q
# compute A+Q**2/2
def eigen_matrix(A,Q):
      return A+.5*np.dot(Q,Q)
# Run through the process for actual data
K_0,K_2,K_4=F_K(Kt(R))
A,Q=AQ(K_0,K_2)
M=eigen_matrix(A,Q)
#####solve eigenvalues,eigenvectors
EIGEN=np.linalg.eig(M)
#the eigenvectors are acually the columns, not the rows!
E=EIGEN[1].transpose() 
# ######select the time coordinates
# time=X[0][0:19]
# test1,test2=E[1],E[2]


#######find your equilibrium datas for each ion
Rscat=Rav.reshape((N,3))
xdata=np.zeros(N)
ydata=np.zeros(N)
zdata=np.zeros(N)
xvect=np.zeros(N)
yvect=np.zeros(N)
zvect=np.zeros(N)
####pick out one eigenvector for plotting
M1=np.real(E[6]).reshape((N,3))

vects=M1
#######put all data in plotting form for quiver plot
for i in range(0,len(Rscat)):
      xdata[i]=Rscat[i,0]
      ydata[i]=Rscat[i,1]
      zdata[i]=Rscat[i,2]
      xvect[i]=vects[i,0]
      yvect[i]=vects[i,1]
      zvect[i]=vects[i,2]
########make a quiver plot      
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim3d(-1,1)

ax.quiver(xdata,ydata,zdata,xvect,yvect,zvect)
ax.scatter(xdata,ydata,zdata)

