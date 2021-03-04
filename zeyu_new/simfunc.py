# -*- coding: utf-8 -*-
"""
@author: Alex Kato, Zeyu Ye
@desc Functions for running the MD simulation.
@date 10/15/2020
"""

import numpy as np
from numba import njit, prange
from userconfig import *
import laser

# laser cooling
###laser wavelength
wav=493e-9
#laser wave number
k_number=2*np.pi/wav

dx=.1
dy=.1
dz=.04
k_vect=1/np.sqrt(dx**2+dy**2+dz**2)*(np.array((dx,dy,dz)))

@njit
def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    """
    vect = np.zeros(3)
    
    phi = np.random.uniform(0, PI * 2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)

    vect[0] = np.sin(theta) * np.cos(phi)
    vect[1] = np.sin(theta) * np.sin(phi)
    vect[2] = np.cos(theta)
    
    return vect

@njit
def rand_v(mu, sigma):
    """
    Random speed generator based on mean and std from Maxwell-Boltzman distribution
    """
    return np.random.normal(mu,sigma)

@njit
def rand_pos():
    """
    Random position generator based on the start area of the simulation.
    """
    return (np.random.random_sample(3) - .5) * start_area

@njit
def initialize_ions(N, hasZ = True):
    """
    Initialize the trajactory of the ions. The last parameter indicates whether it stars on a plane or not.
    """
    
    # Mean and std from Maxwell-Boltzman distribution at given temperature
    mu = np.sqrt(8 * K * T / (PI * m))
    sigma = np.sqrt(K * T / (2 * m))

    init_cdn = np.zeros(6 * N)
    if (hasZ):
        for i in range(1,N+1):
            init_cdn[3*N+3*(i-1):3*N+3*(i-1)+3] = random_three_vector() * rand_v(mu,sigma)      
            init_cdn[3*(i-1):3*(i-1)+3] = rand_pos()
    else:
        for i in range(1,N+1):
            init_cdn[3*N+3*(i-1):3*N+3*(i-1)+3] = random_three_vector() * rand_v(mu,sigma) * np.array([1,1,0])
            init_cdn[3*(i-1):3*(i-1)+3] = rand_pos()* np.array([1,1,0])
    
    return init_cdn

@njit
def coulomb_vect(X, N):
    '''
    takes a vectos of ion positions in Nx1 format and calculates the coulomb force on each ion.
    returns force as Nx1numpy array of force vectors. should inp   
    '''
    F = np.zeros((N,3))
    X3 = X.reshape(N,3)
    # perms=np.array([X3-np.roll(X3,3*i) for i in range(1,N)])
    # F+=kappa*perms/((np.sqrt((perms**2).sum(axis=2)).repeat(3).reshape((N-1,N,3))))**3
    
    for i in prange(1,N):
        #calculate the permutation
        perm=X3-np.roll(X3,3*i)
        #fins the norm of each vector
        # norm=np.sqrt((perm**2).sum(axis=1)).repeat(3).reshape((N,3))

        #calculate colomb forces
        F+=kappa*perm/((np.sqrt((perm**2).sum(axis=1)).repeat(3).reshape((N,3))))**3
        #return flattened array
    return F.ravel()

@njit
def laser_eq(V,N,t):
    # initialize output array
    F=np.zeros((3*N))

    Beta=3.34e-20*(1-t/twindow)**2
    #max is around 5e-21
    if t<=twindow:
        F-=Beta*V
    return F

#####v_old must be a numpy vector 3x1. returns a 3x1 numpy array
@njit
def collision(v_old):
    v_g=rand_v(mu_g,sigma_g)
    n0=random_three_vector() ####random velocity unit vector assigned to ion
    n1=random_three_vector() ####random velocity of gas particle
    return m_g/(m+m_g)*np.abs(np.linalg.norm(v_old)-v_g)*n0+(m*v_old+m_g*v_g*n1)/(m+m_g)
###takes ravelled array of velocities for N ions and updates 

@njit
def collisions(V):
    X=np.zeros(len(V))
    N=int(len(V)/3)
    for i in range(0,N):
        Y=V[3*i:3*i+3]
        X[3*i:3*i+3]=collision(Y)
    return X

@njit
def field_vect(X, N, params):
    
    # Reshape the positions
    indv_pos = X.reshape((N,3))
    
    a,b,c,d,e,f,g = params

    # Initialiize output array
    F = np.zeros((N, 3), dtype=np.float64)
    
    for i in range(0,N):
        x = indv_pos[i,0]
        y = indv_pos[i,1]
        z = indv_pos[i,2]
        F[i,0] = -(2*a*x+d*y+e*z)*Q
        F[i,1] = -(2*b*y+d*x+f*z)*Q
        F[i,2] = -(e*x+f*y+2*c*z)*Q
    
    #print(indv_pos)
    #print(F)

    #Return flattened array
    return F.flatten()

#@njit
def leap_frog(N, init_cdn, DCparams, RFparams, signal):

    # Initialize the time vector
    iterations = int(twindow / tstep)
    time = np.zeros(iterations)
    
    # Initialize trajectories, positions, velocities, accelerations (old([0])/new([1]))
    trajectories = np.zeros((6 * N,iterations), dtype=np.float64)
    trajectories[:,0] = init_cdn
    pos = np.zeros(3 * N, dtype=np.float64)
    vel = np.zeros(3 * N, dtype=np.float64)
    acc = np.zeros((3 * N, 2), dtype=np.float64)
    
    # Initialize the counters for time and for iteration
    it = 1
    t = tstep

    # Assign initial values
    pos = trajectories[0 : 3 * N, 0].copy()
    vel = trajectories[3 * N : 6 * N, 0]
    acc[:, 1] = 1 / m * (coulomb_vect(pos,N) + field_vect(pos, N, DCparams) + +FHYP_pseudo(pos,N))

    # Iterate
    while it < iterations:
        
        # Get current position,velocity of all ions
        acc[:,0] = acc[:,1].copy()
        pos = trajectories[0 : 3 * N, it - 1].copy()
        vel = trajectories[3 * N : 6 * N, it - 1].copy()

        # Update positions based on x'=x+v_i*t+1/2*a*t^2
        trajectories[0 : 3 * N, it] = pos + vel * tstep + .5 * (tstep**2) * acc[:,0].copy()

        # Sum up forces without micromotion
        acc[:,1] = 1 / m * (coulomb_vect(pos,N) + field_vect(pos, N, DCparams) + +FHYP_pseudo(pos,N))# + laser_vect(vel,N))

        # Compute velocities
        trajectories[3 * N : 6 * N, it] = vel + .5 * tstep * (acc[:,0].copy() + acc[:,1].copy())
        trajectories[3 * N : 6 * N, it] = collisions(trajectories[3 * N : 6 * N, it].copy())

        # Update time vector
        time[it] = t
        it += 1
        t += tstep
    
    return time, trajectories

@njit
def FHYP_pseudo(X,N):
      Y=X.reshape((N,3))      #reshaoe the coordinate array to xyz coords of each ion
      # F=np.zeros((N,3))
      ###seperate out the coordinates
      x=Y[:,0]
      y=Y[:,1]
      z=Y[:,2]
      coeff=Q**2*1000**2/(m*omega**2*(.0024**2+2*.001**2)**2)
      R=np.sqrt(x**2+y**2)      #######convert to polar coordinates

      phi=np.arctan2(y,x)
      
      FR=coeff*-2*R
      FZ=-8*coeff*z
      
      F=np.stack((FR*np.cos(phi),FR*np.sin(phi),FZ))
      return F.transpose(1,0).ravel()

@njit
def Newton3(t,X,N, DCparams, RFparams):
    '''
    A few notes about this function
    laser cooling and trap functions are 
    external since they are simpler and faster. 


    The output is all of the derivatives wrt time, for easy input into a diff eq solver like odeint   


    '''
    dt=np.zeros(6*N)#initialize output array

    pos = X[0:3*N]#positions
    vel = X[3*N:6*N]#velocities
    #dt[3*N:6*N]=1 / m * (coulomb_vect(pos,N) + field_vect(pos, N, DCparams) + field_vect(pos, N, RFparams * np.cos(omega*t)) + laser.laser_vect(vel,N) + laser2.laser_vect2(vel,N))
    dt[3*N:6*N]=1 / m * (coulomb_vect(pos,N) + field_vect(pos, N, DCparams) + FHYP_pseudo(pos,N) + laser.laser_vect(vel,N))
                #update v_dot (this is F=ma)
                #######AK you changed this make sure to check it
    dt[0:3*N] = collisions(vel) #vel #update x_dot=v
    #for i in range(0,N):
    #    delta_avg = np.sqrt(laser.vr**2*laser.R*tstep)
    #    delta_spontaneous = delta_avg*laser.random_three_vector()
    #    delta_kick = k_vect * delta_avg
    #    delta = delta_kick+delta_spontaneous
    #    dt[i*3] += delta[0]
    #    dt[i*3+1] += delta[1]
    #    dt[i*3+2] += delta[2]
    return dt

#@njit
def Newton4(t,X,N, DCparams, RFparams):
    '''
    A few notes about this function
    laser cooling and trap functions are 
    external since they are simpler and faster. 


    The output is all of the derivatives wrt time, for easy input into a diff eq solver like odeint   


    '''
    dt=np.zeros(7*N)#initialize output array

    pos = X[0:3*N]#positions
    vel = X[3*N:6*N]#velocities
    excite = X[6*N:7*N]
    #dt[3*N:6*N]=1 / m * (coulomb_vect(pos,N) + field_vect(pos, N, DCparams) + field_vect(pos, N, RFparams * np.cos(omega*t)))
    dt[3*N:6*N]=1 / m * (coulomb_vect(pos,N) + field_vect(pos, N, DCparams) + FHYP_pseudo(pos,N)) #+ laser.laser_vect(vel,N))
                #update v_dot (this is F=ma)
                #######AK you changed this make sure to check it
    dt[0:3*N] = vel #vel #update x_dot=v
    for i in range(0,N):
        if(excite[i] > 0):
            if(excite[i]-tstep>0):
                excite[i] -= tstep
            else:
                excite[i] = 0
                vel[i*3:(i+1)*3] += laser.random_three_vector()*laser.vr
        else:
            kv = np.dot(laser.k_vect,vel[i*3:(i+1)*3])
            # w0=w0+delta+kv -> at the center, so delta+kv=0 is the center of Gaussian
            prob = 1/(laser.sigma*np.sqrt(2*np.pi))*np.exp(-(laser.delta+kv)**2/(2*laser.sigma**2))
            if(prob >= np.random.rand()):
                excite[i] = np.random.exponential(laser.tau)
                vel[i*3:(i+1)*3] += np.linalg.norm(laser.k_vect)*laser.vr
    #for i in range(0,N):
    #    delta_avg = np.sqrt(laser.vr**2*laser.R*tstep)
    #    delta_spontaneous = delta_avg*laser.random_three_vector()
    #    delta_kick = k_vect * delta_avg
    #    delta = delta_kick+delta_spontaneous
    #    dt[i*3] += delta[0]
    #    dt[i*3+1] += delta[1]
    #    dt[i*3+2] += delta[2]
    return dt