# -*- coding: utf-8 -*-
"""
@author: Alex Kato, Zeyu Ye, Samip Karki
@desc Functions for running the MD simulation.
@date 9/3/2021
Modified by Leslie English, June 2022
"""

import numpy as np
from numba import njit, prange
from userconfig import *        # Imports ariables defined in userconfig into simfunc.
import math

# PYTHON NOTES: numpy.load function loads arrays or pickled objects from .npy, .npz or pickled files (per numpy.org)

V04=np.load('VOLT0and4.npy')
V15=np.load('VOLT1and5.npy')
V26=np.load('VOLT2and6.npy')
V37=np.load('VOLT3and7.npy')
V89=np.load('VOLT8and9.npy')

@njit 
# PYTHON NOTES: @jit essentially encompasses two modes of compilation. First it will try and compile the decorated 
# function in 'no Python mode'. If this fails it will try again to compile the function using object mode.
# To make it such that only 'no python mode' is used and if compilation fails an exception is raised, the decorators 
# @njit and @jit(nopython=True) can be used (the first is an alias of the second for convenience).(https://numba.pydata.org/)

def random_three_vector():   # INITIAL ION DIRECTION OF MOTION (random 3-D unit vector direction)
    # CALLED BY: This function is called by the "initialize_ions" function in simfunc.
    # INPUT: none
    # RETURN: A random 3D unit vector (velocity direction) with a uniform spherical distribution
    vect = np.zeros(3) # INPUT: int or tuple of ints. RETURN: Array of given shape and type, filled with zeros. https://numpy.org
    '''
    # Original code  
    phi = np.random.uniform(0, np.pi * 2) # INPUT: low=, high=, RETURN: Samples from uniform distribution over 1/2 open (?) interval [low,high)
    costheta = np.random.uniform(-1, 1)  # INPUT: two values of closed range. RETURN: Random floating point nunmber N in the range between a and b.
    theta = np.arccos(costheta)
    vect[0] = np.sin(theta) * np.cos(phi)
    vect[1] = np.sin(theta) * np.sin(phi)
    vect[2] = np.cos(theta)
    '''
    #Testing to see if initial conditions have much effect on crystal shape
    #phi = np.random.uniform(0, np.pi / 16) # INPUT: low=, high=, RETURN: Samples from uniform distribution over 1/2 open (?) interval [low,high)
    vect[0] = 0
    vect[1] = 0
    vect[2] = 1
    
    return vect

@njit
def rand_v(mu, sigma):    # INITIAL ION SPEED (random scalar value)
#   CALLED BY: This function is called by the "initialize_ions" function in simfunc.
#   INPUT: mu = mean (float) of Maxwell-Boltzmann distribution at a given temperature
#          sigma = standard deviation (float) of Maxwell-Boltzmann distribution at a given temperature
#   RETURN: Random speed (scalar, not velocity vector) from the sampling distribution.

#   << NORMAL DISTRIBUTION >> 
#    print('Random initial speed from Normal Gaussian Distribution with Maxwell-Boltzmann mean and standard deviation.')
#    return np.random.normal(mu,sigma) # INPUT: center, std.dev. RETURN: Random samplesfrom a normal (Gaussian) distribution.
#   PHTHON NOTES: (numpy.random.normal)   random.normal(loc=0.0, scale=1.0, size=None)
#   Draw random samples from a normal (Gaussian) distribution.
#   The probability density function of the normal distribution, first derived by De Moivre and 200 years later by both Gauss and Laplace 
#   independently, is often called the bell curve because of its characteristic shape (see the example below).
#   https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

#   << UNIFORM DISTRIBUTION >>
#    print('Random initial speed from Uniform Distribution spanning (M-B mean - 3sigma) to (M-B mean + 3sigma).')
    return np.random.uniform (mu-3*sigma,mu+3*sigma)  # Uniform distribution spanning -3sigma to +3sigma
#   PHTHON NOTES: random.uniform(a, b)¶
#   Return a random floating point number N such that a <= N <= b for a <= b and b <= N <= a for b < a.
#   The end-point value b may or may not be included in the range depending on floating-point rounding in the equation a + (b-a) * random().
#   https://docs.python.org/3/library/random.html

@njit
def rand_pos():   # INITIAL ION POSITION   (random 3-D location)
    # CALLED BY: This function is called by the "initialize_ions" function in simfunc.
    """
    Random position generator based on the start area of the simulation.
    """
    # PYTHON NOTE: np.random.random_sample(n) returns a 1-D array of n elements taken randomly from [0.0,1.0) (includes 0, not 1)
    # NOTE: variable "start_area" is definted in userconfig.py
    return (np.random.random_sample(3) - .5) * start_area

@njit
def initialize_ions(N, hasZ = True):
    # CALLED BY: This function is called by simmain.
    """
    Initialize the trajactory of the ions. 
    The last parameter indicates whether it starts on a plane or not.
    """
    
    # Mean (mu) and std  from Maxwell-Boltzman distribution at given temperature (T)
    mu = np.sqrt(8 * K * T / (np.pi * m))
    sigma = np.sqrt(K * T / (2 * m))

    init_cdn = np.zeros(6 * N)
    if (hasZ):
        for i in range(1,N+1):  # All 3 direction vector components are multiplied by the same total velocity
            init_cdn[3*N+3*(i-1):3*N+3*(i-1)+3] = random_three_vector() * rand_v(mu,sigma)    
            init_cdn[3*(i-1):3*(i-1)+3] = rand_pos()
    else:
        for i in range(1,N+1):
            init_cdn[3*N+3*(i-1):3*N+3*(i-1)+3] = random_three_vector() * rand_v(mu,sigma) * np.array([1,1,0])
            init_cdn[3*(i-1):3*(i-1)+3] = rand_pos()* np.array([1,1,0])
    
    return init_cdn

@njit
def coulomb_vect(X, N):
    # CALLED BY: This function is called by the "leap_frog3" function in simfunc.
    '''
    takes a vectos of ion positions in Nx1 format and calculates the coulomb force on each ion.
    returns force as Nx1numpy array of force vectors. should inp   
    '''

    F = np.zeros((N,3))
    X3 = X.reshape(N,3)
    
    for i in prange(1,N):
        #calculate the permutation
        perm=X3-np.roll(X3,3*i)
        
        #calculate colomb forces
        F+=kappa*perm/((np.sqrt((perm**2).sum(axis=1)).repeat(3).reshape((N,3))))**3
        
    #return flattened array
    return F.ravel()


@njit

def damp(Dis,V,N,t,T,B=3e-10):

    # CALLED BY: This function is called by the "leap_frog3" function in simfunc.py.
    # PYTHON NOTE:  Non-default input variables must all be listed before default input variables ( B is default, so at end)
    # Uses book "Atomic Physics" by Christopher J. Foot, from Oxford University Press
    # Kato note: uses foote et al paper for reference
    # DRAG FORCE: The laser cooling is modeled as a drag force that is
    #                (1) proportional to current ion velocity, and
    #                (2) exponentially decays over the time span of the simulation
    # This function has been modified to include an alternate model of the laser damping force.
    # P (position of the ions) has been added as an input variable, because in the alternate damping model,
    #    the location of the ion is needed to determine its distance from the laser beam.
    # INPUT: CurrentPositions is (1-D) array of length 3*N of current ion positions
    #      : V is (1-D) array of length 3*N of current ion velocities
    #      : Dist is (1-D) array of distance of each ion from the center of the laser
    #      : t is current time step in the simulation
    #      : T is end time of the simulation
    # OUTPUT: F is (1-D) array of length 3*N of damping force as a function of current ion velocity
  
      ###### REQUIRED FOR BOTH LASER DAMPING OPTIONS #########
      #initialize output arrays
      F=np.zeros((N,3))  # numpy.zeros RETURNS: Array of zeros of given shape. This (2-D) array is N rows x 3 col i.e.: (13,3)
      F=F.ravel()        # numpy.ravel RETURNS: continuous flattened 1-D array. This (1-D) array os N*3 entries. i.e.: (39,)

      # DRAG DAMPING - Force is proportional to velocity
      # (Original From 2021) LASER DAMPING FORCE IS INDEPENDENT OF DISTANCE FROM CENTER OF LASER BEAM.
      Beta=B*(1-(t/T)**2)    # Damping factor Beta tapers off over time so that ion velocity is not driven to zero.
      F-=Beta*V              # Damping force is proportional to current ion velocity   
    
     
      # DRAG DAMPING + LASER FORCE TAPER
      # (New in 2022) LASER STRENGTH TAPERS OFF AS GAUSSIAN FUNCTION OF DISTANCE FROM CENTER OF LASER.
      # This option uses the Beta calculation in Option 1, then supplements it with a tapering Gaussian Factor that 
      # specific for each ion, specifically the distance of each ion from the center of the laser.
      # This Gaussian Factor is not calculated from the Optics Gaussian Beam, which is dependent upon the laser wavelength,
      # but rather from a generic Gaussian curve centered at zero, max height one, defined only by standard deviation.
      
      # Create and initialize arrays to store Gaussian laser radial taper factor
      GaussianFactor = np.zeros(N, dtype=np.float64)
      # Calculate laser strength taper factor for each ion.
      GaussianFactor = np.exp(  - (Dis/StdDev)**2 )
      GF = np.zeros(3*N)   
      for i in range(N):
          GF[3*i] = GaussianFactor[i]
          GF[3*i+1] = GaussianFactor[i]
          GF[3*i+2] = GaussianFactor[i]

      # Multiply damping force on each ion (which is proportional to the ion's velocity) by its Gaussian Factor '''
      #F *= GF # This is the only line that must be commented out in order to run only option 1.
      # CHOOSE one of these two print messages to document which laser model was used.
        
      ###### REQUIRED FOR BOTH LASER DAMPING OPTIONS ##############
      return F


@njit
def Displace(N,wx=243e3,wy=147e3,wz=768e3,dx=-9.87e-6,dy=-1.85e-6,dz=0):
    # CALLED BY: This function is called by the "leap_frog3" function in simfunc.
    Fx=-m*(wx*2*np.pi)**2*dx
    Fy=-m*(wy*2*np.pi)**2*dy
    Fz=-m*(wz*2*np.pi)**2*dz
    F=np.array([Fx,Fy,Fz]*N)
    return F

@njit
def field_vect(X, N, params):
    # CALLED BY: This function is called by the "leap_frog3" function in simfunc.
    # Reshape the positions
    # PYTHON NOTES: numpy.reshape(array,newshape,order) Gives a new shape to an array without changing its data.
    #    newshape should be compatible with the original shape: int or tuple of ints
    #    If newshape is an integer, then the result will be a 1-D array of that length.
    indv_pos = X.reshape((N,3))
    
    a,b,c,d,e,f,g = params

    # Initialize output array
    F = np.zeros((N, 3), dtype=np.float64)
    
    for i in range(0,N):
        x = indv_pos[i,0]
        y = indv_pos[i,1]
        z = indv_pos[i,2]
        
        F[i,0] = -(2*a*x+d*y+e*z)*Q
        F[i,1] = -(2*b*y+d*x+f*z)*Q
        F[i,2] = -(e*x+f*y+2*c*z)*Q

    #Return flattened array
    return F.flatten()

@njit
def leap_frog3(N, init_cdn, Side_Stretch_V, Side_Pinch_V, Endcap_V, RFamp, tstart, twindow, accf):
    # CALLED BY: This function is called by simmain file.
    # NOTE: The incoming variable accf is actually the initial acceleration (accf=0) from simmain.
    # OUTPUT: A tuple of the following arrays: time, trajectories, accf, DistanceData
    # Initialize the time vector
    iterations = int(twindow / tstep)
    B=3e-20
    T=twindow+tstart   # End time of the simulation. Local variable. Not equal to temperature.
    time = np.ones(iterations)*tstart   # Return a new array of given shape and type, filled with ones.
    print('LEAPFROG FUNCTION')
    print()
    print('    Time Window = ', twindow)
    print('    Timestep = ', tstep)    
    print('    Number of iterations = ', iterations)
    print('    T = twindow + tstart = ', T)
    print()
    print('    Number of ions = ', N)
    print()
    
    # Create & Initialize arrays to ZERO for trajectories, positions, distances, velocities, accelerations (old([0])/new([1]))
    # PYTHON NOTES:  class numpy.ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)[source]:
    #    An array object represents a multidimensional, homogeneous array of fixed-size items. 
    #    An associated data-type object describes the format of each element in the array (its byte-order, 
    #    how many bytes it occupies in memory, whether it is an integer, a floating point number, or something else, etc.)
    #    Arrays should be constructed using array, zeros or empty (refer to the See Also section below). 
    #The parameters given here refer to a low-level method (ndarray(…)) for instantiating an array.
    trajectories = np.zeros((6 * N,iterations), dtype=np.float64) # 2-D array for ALL position & velocity data for all iterations.
    pos = np.zeros(3 * N, dtype=np.float64)                       # 1-D array, CURRENT x, y, z pos values for all ions
    vel = np.zeros(3 * N, dtype=np.float64)                       # 1-D array, CURRENT x, y, z vel values for all ions
    acci= np.zeros(3 * N, dtype=np.float64)                       # 1-D array
    DistanceData = np.zeros((N, iterations), dtype=np.float64)   # 2-D array for ALL distances from laser & from center for each ion.
    DistL = np.zeros(N, dtype=np.float64)                      # 1-D array, CURRENT distance from laser center for all ions
    
    # Assign initial position & velocity values (RANDOMLY GENERATED BY OTHER FUNCTIONS) for 1st iteration.
    trajectories[:,0] = init_cdn
    pos = trajectories[0 : 3 * N, 0].copy()
    vel = trajectories[3 * N : 6 * N, 0].copy()    

    # Initialize the counters for time and for iteration
    t = tstart    
    it=1             # 1st Integration step
    t+=tstep         # Time = Tstart + Tstep
    time[it] = t  

    while it < iterations:
        
        # Get CURRENT (from prior iteration) position, velocity of all ions
        pos = trajectories[0 : 3 * N, it - 1].copy()
        vel = trajectories[3 * N : 6 * N, it - 1].copy()
      
    
        # Sum up forces without micromotion
        # Update starting acceleration for each iteration to be same as ending acceleration of prior iteration.
        acci=accf.copy()
               
        # Update positions based on x'=x+vt+at^2
        trajectories[0 : 3 * N, it] = pos + vel*tstep +(1/2)*acci*tstep**2
        pos= trajectories[0 : 3 * N, it].copy()
       
        # Calculate UPDATED distance of each ion from center of laser
        DistL = np.sqrt ( (0)**2  +  (pos[1::3])**2  + ( pos[2::3] - (math.tan(LaserRadians))*(pos[0::3]) )**2 )

        ######### CALCULATE NET FORCE IN THE TRAP  ###########
        #
        # Net of All Forces
        #   coulomb_vect    = Coulomb repulsion force between ions (all +1). Repulsion force depends only upon distance between ions.
        #   Side_Stretch_V  = multiplication factor to scale side ring sector DC voltage (defaulted to 1 volt) in the stretch direction
        #   Side_Pinch_V    = multiplication factor to scale side ring sector Dc voltage in the pinch direction
        #   Endcap_V        = multiplication factor to scale the endcap DC voltage (electrodes 8 & 9)
        #   RFamp           = multiplication factor to scale the endcap AC voltage
        #   field_vect      = function that calculates simulated force based upon specific configuration of electrode voltages
        #   Damp            = Damping drag force, proportional to current velocity of the ions
        #   Displace        = Displacement force for off-set from center of trap.
        #
        # Do not change anything in the following equation. Change variable values in "userconfig.py"
        all_forces = \
        coulomb_vect(pos,N) + \
        Factor04 * field_vect(pos, N, V04) + \
        Factor15 * field_vect(pos, N, V15) + \
        Factor26 * field_vect(pos, N, V26) + \
        Factor37 * field_vect(pos, N, V37) + \
        Endcap_V * field_vect(pos, N, V89) + \
        RFamp*field_vect(pos, N, V89) * np.cos(omega*t) + \
        damp(DistL,vel,N,t,T,B) + \
        Offset * Displace(N)
          
        ####### CALCULATE ACCELERATION from Newton's Law F=ma #########
        accf = all_forces/m    # Only accf from the final iteration is returned. No other accf are stored.
        
        # Store x,y,z velocities and distance data from current time step.
        trajectories[3 * N : 6 * N, it] = vel + tstep * ((acci+accf)/2)
        DistanceData[:,it] = DistL
        
        # Update time vector
        time[it] = t
        it += 1
        t += tstep
        
    print('LEAPFROG IS COMPLETE.')
    print()
    return time, trajectories, accf, DistanceData
        