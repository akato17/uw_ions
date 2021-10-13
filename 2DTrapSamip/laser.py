# -*- coding: utf-8 -*-
"""
@author: Alex Kato, Zeyu Ye
@desc Laser vector and paramters.
@date 9/16/2020
"""
import numpy as np
from mdconst import *
from numba import njit, prange
import userconfig

m = 138*M

####excited state lifetime for p1/2 state
tau=8.1e-9 ####I need an actual referenece for this
####gamma
gamma=1/tau


#laser direction
dx=1
dy=0
#dy=0
dz=-0.172793
#dz=0


'''Frequencies'''
#atom wavelength and freq
atwav=493e-9#
atomfreq=C/atwav#
watom=atomfreq*2*np.pi#

#laser parameters
wlaser= watom-0.5*gamma
# wlaser=watom-6.3724e8
lasfreq=wlaser/(2*np.pi)
laswav=C/lasfreq


#laser wave number
k_number=2*np.pi/laswav
#k_number=2*np.pi/atwav
#k magnitude and vector
k_vect=1/np.sqrt(dx**2+dy**2+dz**2)*(np.array((dx,dy,dz)))
K=k_vect*k_number


###saturation intensity
Isat=np.pi*H*C/(3*atwav**2*tau)
####laser intensity
I=2*Isat


##delta i.e. detuning

delta=wlaser-watom #wlaser-watom

#Doppler brodening
FWHM = gamma*np.sqrt(1+I/Isat)
sigma = FWHM/2.355

###saturation parameter
s0=I/Isat #this is maximum cooling parameter
# s0=0


###lorentzian factor
S=s0/(1+(2*delta/gamma)**2)

#Recoil velocity and scattering rate
vr = hbar*k_number/userconfig.m
rabi_square = I/Isat/2*gamma**2
R = gamma/2*(rabi_square/2)/(delta**2+rabi_square/2+gamma**2/4)


# A copy of random functions for numba
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
    time=0
    #initialize output array
    F=np.zeros((N,3), dtype=np.float64)
    vel=V.reshape((N,3))
    
    #scattering force
    F0=hbar*K*gamma*S/2/(1+S)
    # F0=0
    F+=F0
    #flatten array
    F=F.ravel()

    ###damping coefficient
    Beta=-hbar*4*s0*delta/gamma/(1+s0+(2*delta/gamma)**2)**2*np.kron(vel.dot(K),K)
    F-=Beta
    
    # if time > twindow/2:
    #     F=F*0

    return F
'''Discrete Laser'''
#@njit
# def poisson_process(mu): #input mean, mu of poisson process, get back how many occurances for a small mu, 0,1, or 2 occurances
#     emissions=0
#     zero=(mu**0)*np.exp(-mu)/np.math.factorial(0)
#     one=(mu**1)*np.exp(-mu)/np.math.factorial(1)
#     #two=(mu**2)*np.exp(-mu)/np.math.factorial(2)
#     rand=np.random.uniform()
#     if rand>zero :
#         emissions=1
#     else:
#         emissions=0
#     return emissions
    
#     emissions=np.random.poisson(mu)
'''This only works well if the ions are already pretty cool, so have cold initial conditions'''
@njit
def Rscatt(Vel): #takes in single ion velocity and finds the Rscatt, based on Foot book, this works fine
    kv=-1*(Vel[0]*K[0]+Vel[1]*K[1]+Vel[2]*K[2])#
    #Rate= (gamma/2)*(I/Isat)/(1+(I/Isat)+(4*delta**2/gamma**2))
    Rate = (s0*gamma/2)/(1+s0+(2*(delta+kv)/gamma)**2)
    return Rate
@njit
def discrete_laserF(V,N):#want to take in velocity and output laser force
    F=np.zeros((N,3), dtype=np.float64)
    vel=V.reshape((N,3))
    for i in range(0,N):
        mu=Rscatt(vel[i])#find scattering rate based on K.V
        mu=mu*userconfig.tstep #now it is scaled properly, for Rscatt to be mu of poisson distribution with tstep as time, resize scattering rate so we have emissions/tstep
        #run poisson dist, getting 0,1,2 emissions
        emissions=np.random.poisson(mu)
        for j in range(0,emissions):
            F[i]+= (hbar*K+hbar*k_number*random_three_vector())*(1/userconfig.tstep)#both forces, emission and absorption
        #add up forces on that one ion
        #append the force into F
    return F.ravel()

@njit
def discrete_laserVr(V,N):#want to take in velocity and output kick velocity
    vkick=np.zeros((N,3), dtype=np.float64)
    vel=V.reshape((N,3))
    for i in range(0,N):
        mu=Rscatt(vel[i])#find scattering rate based on K.V
        mu=mu*userconfig.tstep #now it is scaled properly, for Rscatt to be mu of poisson distribution with tstep as time, resize scattering rate so we have emissions/tstep
        #run poisson dist, getting 0,1,2 emissions
        emissions=np.random.poisson(mu)
        for j in range(0,emissions):
            vkick[i]+= (vr*k_vect+vr*random_three_vector())#both forces, emission and absorption
        #add up forces on that one ion
        #append the force into F
    return vkick.ravel()

'''splitting "discrete_laser" into two steps, absorption and emission'''
@njit
def twosteplaser(V,N,excitations):
    vkick=np.zeros((N,3), dtype=np.float64)
    vel=V.reshape((N,3))
    for i in range(0,N):
        if excitations[i]==0:#if ion is in ground state, see if it absorobed
            scatter= Rscatt(vel[i])
            prob= (scatter/(1-tau*scatter))*userconfig.tstep
            #prob= scatter*userconfig.tstep --> pretty sure this is wrong probability
            randn=np.random.uniform(0,1)
            if randn<prob:
                vkick[i]+=vr*k_vect
                excitations[i]=1
                
        else:#if ion is in excited state, see if it emits
            prob=gamma*userconfig.tstep
            randn=np.random.uniform(0,1)
            if randn<prob:
                vkick[i]+=vr*random_three_vector()
                excitations[i]=0
                
    return vkick.ravel(), excitations

