# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:18:16 2020

@author: barium
this module is written to calculate the normal modes of oscillation. It assumes youalready 
have the equilibrium trajectories



NOTE: this is based on the papr "normal modes of oscillation in paul traps" by Landa, but follow analysis 
in the thesis https://deepblue.lib.umich.edu/bitstream/handle/2027.42/149953/wyukai_1.pdf?sequence=1
from Luming Duan's group, it is much simpler'
"""
import numpy as np
from tkinter.filedialog import askopenfilename
import tkinter
from numba import njit
from matplotlib import pyplot as plt
from scipy.constants import e,physical_constants,epsilon_0
from PyQt5.QtWidgets import QFileDialog
import scipy.optimize as optimize

# 
#####open file dialog. Choose a trajectory data file or this wont work
####alternatively, manually type in filepath to np.load command
# root=tkinter.Tk()
# filename = askopenfilename()
# root.destroy()
filename=QFileDialog.getOpenFileName()[0]

# load the file
Y=np.load(filename,allow_pickle=True)
#####must use this line ifusing samip's file format
# X=np.array((0,Y['Y']))
####or this line if old format
X=Y
#####decipher how many ions were in the simulation
N=int(len(X[1])/6)
########mass of barium
M=physical_constants['atomic mass constant'][0]
mb=138*M
#####rf drive frequency
omega=2*np.pi*10.42e6
####length scale to make data dimensionless
L=((e**2)/(4*np.pi*epsilon_0*mb*omega**2))**(1/3)
###R must be dimensionless
Rt=np.transpose(X[1][0:3*N])[0]/(L)
# time=np.linspace(0,len(Rt),1)
time=Y[0][-20:]*omega/2
period=len(time)

###pick out the eq positions over one period of micromotion (omega)
traj=X[1][0:3*N].transpose()[-20:].transpose()
# time=np.linspace(0,len(traj[0])-1,len(traj[0]))
R=traj/L

# R=traj.transpose()/L

###pick out the eq positions over one period of micromotion
# traj=X[1][17][0:19]
########input all matthieu paramaters. soon to be automatic from trajectory data
r_0=2e-3
z_0=.5e-3
q_z=-8*e*2000/(mb*np.sqrt(r_0**2+2*z_0**2)*omega**2)
q_x=(0.031+0.028)/2
q_x=-q_x
# q_x=-q_z/2
# q_y=-q_z/2
q_y=q_x
q_z=-2*q_x
Vend=7
r0=2e-3
z0=.5e-3
a_z=-16*e*Vend/((mb*np.sqrt(r0**2+2*z0**2))*omega**2)
a_z=a_z*.5
a_x=-a_z/2
a_y=-a_z/2

a=np.array([a_z,a_y,a_z])
q=np.array([q_x,q_y,q_z])
#####average the ion's position over one micromotion period
Rav=np.average(R,axis=1)
##########sum functions just to make the K matrix from the paper
def sum1(R,i,j):
    A=np.zeros(len(time))
    A=(R[3*i]-R[3*j])**2+(R[3*i+1]-R[3*j+1])**2+(R[3*i+2]-R[3*j+2])**2
    return A
def sum2(R,i,sigma,tau):
      A=np.zeros(len(time))
      for k in range(0,int(len(R)/3)):
                  
            
                  
                  if k==i:
                        continue
                  else:
                        
                        A+=(R[3*i+sigma]-R[3*k+sigma])*(R[3*i+tau]-R[3*k+tau])/sum1(R,i,k)**(5/2)
                        
      return A
def sum3(R,i,sigma):
      A=np.zeros(len(time))
      for k in range(0,int(len(R)/3)):
            
            if k==i:
                  continue
            else:
                  A+=(sum1(R,i,k)-3*(R[3*i+sigma]-R[3*k+sigma])**2)/sum1(R,i,k)**(5/2)
      return A
#########compute the K matrix as in Yukai thesis
def K(R):
      '''
      
takes an input vector of 3N ion positions and spits out the K matrix for those positions
      returns a 3Nx3N matrix for each point in time
      

      '''
      N=int(len(R)/3)
      K=np.zeros((3*N,3*N,len(time)))
      for k in range(0,len(R)):
            for l in range(0,len(R)):
                  sigma=k%3
                  tau=l%3
                  i=int(k/3)
                  j=int(l/3)
                  # print(i,j,sigma,tau)

                  if i!=j and sigma!=tau:
                        K[:][k][l]=-3*(R[3*i+sigma]-R[3*j+sigma])*(R[3*i+tau]-R[3*j+tau])/\
                              sum1(R,i,j)**(5/2)
                        # print("yes")
                  elif i!=j and sigma==tau:
                        K[:][k][l]=(sum1(R,i,j)-3*(R[3*i+sigma]-R[3*j+sigma])**2)/\
                              sum1(R,i,j)**(5/2)
                        # print("yes")
                  elif i==j and sigma!=tau:
                        K[:][k][l]=3*sum2(R,i,sigma,tau)
                        # print("yes")
                  elif i==j and sigma==tau:
                        K[:][k][l]=-1*sum3(R,i,sigma)
                        # print("yes")
      return K
def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag

# #####compute the K matrix for each point over the period of micromotion
# def Kt(R):
#       Q=np.zeros((len(R),3*N,3*N))
#       for i in range(0,len(R)):
#             Q[i]=K(R[i])
            
#       return Q
def Fit_Cosine(Data,cords,plot=False,ionno=''):
	om=2*np.pi*(1/(len(cords)+1))
	def Cos_Curve(t, a, offset, phase):
		return a*np.cos(2*t+phase) + offset
# 	fitdata=Data.ravel(
# 	guess_omega=2*np.pi*(1/(len(cords)+1)
	guess_amp=(np.max(Data)-np.min(Data))/2

	
    
	guess_phase=0.05
	guess_offset=(np.max(Data)+np.min(Data))/2
	

	guess=(guess_amp,guess_offset,guess_phase)

	err=np.ones(len(cords))*.01
	params,cov = optimize.curve_fit(Cos_Curve, cords, Data,p0=guess,maxfev=100000,sigma=err,absolute_sigma=True)

	return params[0],params[1]
#######compiute the first two fourier components of the K matrix over one period
def F_K(K):
      """
      computes the fourier decomposition of K(t)
      assuming one period of data. must change if more
      """
      # A=np.fft.fftn(K,axes=[2])
      # a0=np.real(A[0])
      # a1=np.real(A[1]+A[-1])
      # a2=np.real(A[2]+A[-2])
      a0=np.zeros((3*N,3*N))
      a1=np.zeros((3*N,3*N))
      # a2=np.zeros((3*N,3*N))
      # b1=np.zeros((3*N,3*N))
      # b2=np.zeros((3*N,3*N))
      # A=np.fft.fftn(K,axes=[0])
      # a0=np.real(A[:,:,0])
      # a1=np.real(A[:,:,1]+A[:,:,-1])
      # a2=np.real(A[:,:,2]+A[:,:,2])
      
      for i in range(0,len(K)):
           for j in range(0,len(K)):
               amp,offset=Fit_Cosine(K[:][i][j],time)
               # fseries=fourier_series_coeff_numpy(K[:][i][j], (2*np.pi/omega)*omega/2, 1, return_complex=False)
               # k0=fseries[0]
               # k1=fseries[1][0]
               k0=offset
               k1=amp
               a0[i][j]=k0
               a1[i][j]=k1
               print(i,j)
      # print(fseries)
      return a0,a1
def F_K_Test(K0,K1,t):
    A=np.zeros((3*N,3*N,len(t)))
    for i in range(0,len(t)):
        A[:,:,i]+=K1*np.cos(2*t[i])
        # A[:,:,i]=A[:,:,i]*np.cos(2*t[i])
        A[:,:,i]=A[:,:,i]+K0
    # A+=K_0
    return A



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
      return A+.5*np.linalg.matrix_power(Q,2)
# Run through the process for actual data
K_0,K_2=F_K(K(R))
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

ax.quiver(xdata,ydata,zdata,xvect,yvect,zvect,linewidths=2)
ax.scatter(xdata,ydata,zdata)
