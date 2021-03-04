# -*- coding: utf-8 -*-
"""
@author: Alex Kato
@desc Introduce all simulation related constants for import.
@date 9/16/2020
"""
import numpy as np
from mdconst import *
from numba import njit, prange
import userconfig

# laser cooling
###laser wavelength
wav=493e-9
#laser wave number
k_number=2*np.pi/wav

dx=.1
dy=.1
dz=.04
k_vect=1/np.sqrt(dx**2+dy**2+dz**2)*(np.array((dx,dy,dz)))
K=k_vect*k_number
###laser frequency
freq=C/wav
##delta i.e. detuning
####excited state lifetime for p1/2 state
tau=8.1e-9 ####I need an actual referenece for this
####gamma
gamma=1/tau
###saturation intensity
Isat=np.pi*H*C/(3*wav**2*tau)
####laser intensity
###optimal is 2Isat
I=3*Isat
I=2*Isat
# I=0
delta=-0.5*gamma
###optimal is -0.5gamma

FWHM = gamma*np.sqrt(1+I/Isat)
sigma = FWHM/2.355

###saturation parameter
s0=I/Isat
# s0=0
###center frequency
w_0=freq*2*np.pi
###lorentzian factor
S=s0/(1+(2*delta/gamma)**2)
# Beta=-hbar*k_number**2*4*s0*delta/gamma/(1+s0+(2*delta/gamma)**2)**2

vr = hbar*k_number/userconfig.m
rabi_square = I/Isat/2*gamma**2

R = gamma/2*(rabi_square/2)/(delta**2+rabi_square/2+gamma**2/4)

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
def laser_vect(V,N):
    '''
    uses foote et al paper for reference

    '''
    #initialize output array
    F=np.zeros((N,3), dtype=np.float64)
    vel=V.reshape((N,3))
    # delta=L[1]
    # s0=L[2]
    # K=L[3]
    # s0=.25
    
    # F0=0
    ###project velocity into laser direction
    # speedk=-vel.dot(k_vect)
    # delta=-200e6*2*np.pi+k_number*speedk*0
    # delta=0
    S=s0/(1+(2*delta/gamma)**2)
    # S=np.zeros((N,1))
        #####################leave out F0 for now

    F0=hbar*K*gamma*S/2/(1+S)
    # F0=0
    F+=F0
    #################################
    #calculate recoil
    #F+=hbar*S/(1+S)*(.5*gamma*K)

    #flatten array
    # F+=np.kron(S,k_vect)
    F=F.ravel()
    ###damping coefficient
    # F.ravel()
    Beta=-hbar*4*s0*delta/gamma/(1+s0+(2*delta/gamma)**2)**2*np.kron(vel.dot(K),K)
    # Beta=8e-21
    # Beta=-5e-22
    # Beta=0
    # F-=Beta*np.kron(vel.dot(k_vect),k_vect)
    F-=Beta

    return F