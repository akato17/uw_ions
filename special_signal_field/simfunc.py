# -*- coding: utf-8 -*-
"""
@author: Alex Kato, Zeyu Ye
@desc Functions for running the MD simulation.
@date 10/15/2020
"""

import numpy as np
from numba import njit, prange
from userconfig import *

#@njit
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

#@njit
def rand_v(mu, sigma):
    """
    Random speed generator based on mean and std from Maxwell-Boltzman distribution
    """
    return np.random.normal(mu,sigma)

#@njit
def rand_pos():
    """
    Random position generator based on the start area of the simulation.
    """
    return (np.random.random_sample(3) - .5) * start_area

#@njit
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

#@njit
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

#####v_old must be a numpy vector 3x1. returns a 3x1 numpy array
#@njit
def collision(v_old):
    v_g=rand_v(mu_g,sigma_g)
    n0=random_three_vector() ####random velocity unit vector assigned to ion
    n1=random_three_vector() ####random velocity of gas particle
    return m_g/(m+m_g)*np.abs(np.linalg.norm(v_old)-v_g)*n0+(m*v_old+m_g*v_g*n1)/(m+m_g)
###takes ravelled array of velocities for N ions and updates 

#@njit
def collisions(V):
    X=np.zeros(len(V))
    N=int(len(V)/3)
    for i in range(0,N):
        Y=V[3*i:3*i+3]
        X[3*i:3*i+3]=collision(Y)
    return X

#@njit
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
    acc[:, 1] = 1 / m * (coulomb_vect(pos,N) + field_vect(pos, N, DCparams) + field_vect(pos, N, RFparams * signal[0] / 1000))

    # Iterate
    while it < iterations:
        
        # Get current position,velocity of all ions
        acc[:,0] = acc[:,1].copy()
        pos = trajectories[0 : 3 * N, it - 1].copy()
        vel = trajectories[3 * N : 6 * N, it - 1].copy()
        
        # Get current RF field
        t_index = int((t - int(t/period)*period) / sample_step) # Position within a period

        # Update positions based on x'=x+v_i*t+1/2*a*t^2
        trajectories[0 : 3 * N, it] = pos + vel * tstep + .5 * (tstep**2) * acc[:,0].copy()

        # Sum up forces without micromotion
        acc[:,1] = 1 / m * (coulomb_vect(pos,N) + field_vect(pos, N, DCparams) + field_vect(pos, N, RFparams * signal[int((t - period * int(t / period)) / sample_step)] / 1000) + laser_vect(vel,N))

        # Compute velocities
        trajectories[3 * N : 6 * N, it] = vel + .5 * tstep * (acc[:,0].copy() + acc[:,1].copy())
        trajectories[3 * N : 6 * N, it] = collisions(trajectories[3 * N : 6 * N, it].copy())

        # Update time vector
        time[it] = t
        it += 1
        t += tstep
    
    return time, trajectories