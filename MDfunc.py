# -*- coding: utf-8 -*-
"""


@author: Alex Kato
"""
import numpy as np
import mdconst as mc
from numba import njit, prange



"""
Input constants here
"""

####constant for calculating the coulomb force
kappa=mc.e**2/(4*np.pi*mc.epsilon_0)

###laser wavelength
wav=493e-9
#laser wave number
k_number=2*np.pi/wav
###wave vector
# k_vect=1/np.sqrt(3)*np.array((1,1,1))
###note to self 2:3:1, 1:1:1 work, 1:1:0.3
######################set the direction of laser cooling
dx=.1
dy=.1
dz=.04
k_vect=1/np.sqrt(dx**2+dy**2+dz**2)*(np.array((dx,dy,dz)))
K=k_vect*k_number
###laser frequency
freq=mc.c/wav
##delta i.e. detuning
####excited state lifetime for p1/2 state
tau=8.1e-9 ####I need an actual referenece for this
####gamma
gamma=1/tau
###saturation intensity
Isat=np.pi*mc.h*mc.c/(3*wav**2*tau)
####laser intensity
###optimal is 2Isat
I=3*Isat
I=2*Isat
# I=0
delta=-.5*gamma
delta=-0.5*gamma
###optimal is -0.5gamma

###saturation parameter
s0=I/Isat
# s0=0
###center frequency
w_0=freq*2*np.pi
###lorentzian factor
S=s0/(1+(2*delta/gamma)**2)
# Beta=-hbar*k_number**2*4*s0*delta/gamma/(1+s0+(2*delta/gamma)**2)**2
"""
Trap parameters
"""
####secular frequencies if using no micromotion
# wx=.205e6*2*np.pi
# wy=.22e6*2*np.pi
# wz=.6e6*2*np.pi
#####hyperbolic trap parameters
r0=.002
z0=.0005
V=800
####rf drive frequency
# omega=10e6*2*np.pi 
omega=12.46e6*2*np.pi

"""
Magnetic field
"""
B=1e-3*np.array([0,0,1])
"""
initial conditions parameters
"""
### initial conditions of barium ions, temperature, boltzman mean and std
Tb=500
# mu_barium=np.sqrt(8*k*Tb/(np.pi*mb))
# sigma_barium=np.sqrt(k*Tb/(2*mb))
###temperature of virtual gas

###size of grid where initial positions may start
start_area=100e-6

"""
if using collisions method of damping
"""
####mass of cold gas for ccd reproduction images
mg=mc.mb/100
T=2e-3
###mean and std of boltzmann distribution for virtual gas
mu=np.sqrt(8*mc.k*T/(np.pi*mg))
sigma=np.sqrt(mc.k*T/(2*mg))



#####initial conditions

"""
Fitted functions for ac and dc potentials that yield a force on a charged particle
"""
# 
@njit
def stray_field(N,t):
      '''
      simple function that returns a force due to stray dc field
      '''
      # start=fstart*2*np.pi
      # slope=fsweep*2*np.pi/tsweep
      # freq=slope*(t-tSTART)+start
      F=np.zeros(3*N)
      E=10
      F-= mc.e*E*np.cos(freq*t)
      return  F

@njit
def FHYP_pseudo(X,N):
      Y=X.reshape((N,3))      #reshaoe the coordinate array to xyz coords of each ion
      # F=np.zeros((N,3))
      ###seperate out the coordinates
      x=Y[:,0]
      y=Y[:,1]
      z=Y[:,2]
      coeff=mc.e**2*V**2/(mc.mb*omega**2*(r0**2+2*z0**2)**2)
      R=np.sqrt(x**2+y**2)      #######convert to polar coordinates

      phi=np.arctan2(y,x)
      
      FR=coeff*-2*R
      FZ=-8*coeff*z
      
      F=np.stack((FR*np.cos(phi),FR*np.sin(phi),FZ))
      return F.transpose(1,0).ravel()
      
      
 
@njit
def FHYP_vect(X,N,t):
      """
      this kinf of force is heavily dependent on the coordinate system. Therefore
      we first convert to polars and seperate ions by coordinates. The forces are then 
      computed and reassigned to the ions
      X is array of 3N positions in flat array, N is number of ions, t is time
      """
      coeff=2*mc.e*V/(r0**2+2*z0**2)*np.cos(omega*t) #calculate the coefficient for the force for a hyperbolic ion trap
      
      Y=X.reshape((N,3))      #reshaoe the coordinate array to xyz coords of each ion

      ###seperate out the coordinates
      x=Y[:,0]
      y=Y[:,1]
      z=Y[:,2]
      
      R=np.sqrt(x**2+y**2)      #######convert to polar coordinates

      phi=np.arctan2(y,x)
      
      FR=-coeff*R       #calculate the radial trap force

      #restack the array in cartesian coordinates including z force
      
      F=np.stack((FR*np.cos(phi),FR*np.sin(phi),2*coeff*z))
      #return output as flattened array
      return F.transpose(1,0).ravel()
@njit
def F_DC(X,N):
      """
      
creates fake dc force (to be replaced by real one soon)
somehow setting values too high can cause simulation not to work

      """
      ####reshape input array
      A=X.reshape((N,3))
      ###initialiize output array
      F=np.zeros((N,3))
      ####choose amount to increase/decrease forces by
      wx=-.14e6*2*np.pi
      wy=.14e6*2*np.pi
      wz=.8e6*np.pi
      #set order of forces
      trap=np.array([wx,wy,wz])
      #set sign of forces
      order=np.array([1,-1,1])
      #calculate force
      F-=mc.mb*trap**2*order*A
      #return flattened array
      return F.ravel()
      
      
@njit
def F_DC_fake(X,N,t):
      ####reshape input array
      A=X.reshape((N,3))
      FFF=1e3*2*np.pi
      ###initialiize output array
      F=np.zeros((N,3))
      ####choose amount to increase/decrease forces by
      wx=0e6*2*np.pi*np.cos(FFF*t)
      wy=.2e6*2*np.pi*np.sin(FFF*t)
      wz=2.0e6*np.pi
      #set order of forces
      trap=np.array([wx,wy,wz])
      #set sign of forces
      order=np.array([1,1,1])
      #calculate force
      F-=mc.mb*trap**2*order*A
      #return flattened array
      return F.ravel()





      
@njit
def laser_vect(V,N):
      '''
      uses foote et al paper for reference

      '''
      #initialize output array
      F=np.zeros((N,3))
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

      F0=mc.hbar*K*gamma*S/2/(1+S)
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
      Beta=-mc.hbar*4*s0*delta/gamma/(1+s0+(2*delta/gamma)**2)**2*np.kron(vel.dot(K),K)
      # Beta=8e-21
      # Beta=-5e-22
      # Beta=0
      # F-=Beta*np.kron(vel.dot(k_vect),k_vect)
      F-=Beta
      return F
'''laser sweep function comment out for now
@njit
def laser_sweep(V,N,t):
      # initialize output array
      start=fstart*2*np.pi
      slope=fsweep*2*np.pi/tsweep
      # s0=.25
      freq=slope*(t-tSTART)+start
      s0=.25*I/Isat*(3+np.cos(freq*t))
      delta=-.5*gamma*(1+.5*np.cos(freq*t))
      S=s0/(1+(2*delta/gamma)**2)

      F=np.zeros((N,3))
      
      vel=V.reshape((N,3))
      # F0=0
      ###project velocity into laser direction
      # speedk=-vel.dot(k_vect)
      # delta=-200e6*2*np.pi+k_number*speedk*0
      # delta=0
      # S=s0/(1+(2*delta/gamma)**2+s0)*hbar*k_number*gamma/2
      # S=np.zeros((N,1))
      F0=hbar*K*gamma/2*S/(1+S)
      F+=F0
      #calculate recoil
      #F+=hbar*S/(1+S)*(.5*gamma*K)
      
      #flatten array
      # F+=np.kron(S,k_vect)
      F=F.ravel()
      ###damping coefficient
      # F.ravel()
      Beta=-hbar*4*s0*delta/gamma/(1+s0+(2*delta/gamma)**2)**2*np.kron(vel.dot(K),K)
      # Beta=5e-21
      # Beta=-5e-22
      # Beta=0
      # F-=Beta*np.kron(vel.dot(k_vect),k_vect)
      F-=Beta
      return F
'''
      
@njit
def laser_eq(V,N,t):
         # initialize output array
      
      

      F=np.zeros((3*N))
      bigT=.004
      
      Beta=3.34e-20*(1-t/bigT)**2
      #max is around 5e-21
      if t<=bigT:
            F-=Beta*V
      return F  
@njit
def Coulomb_vect(X,N):
      '''
   takes a vectos of ion positions in Nx1 format and calculates the coulomb force on each ion.
   returns force as Nx1numpy array of force vectors. should inp   
      
   '''
      # F=np.zeros((N-1,N,3))
      F=np.zeros((N,3))
      X3=X.reshape(N,3)
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
def Coulomb_jit(X,N):
    F=np.zeros((N,3)) 
    x=X.reshape(N,3)
    for i in prange(0,N):
                for j in prange (0,N):
                      if i==j:
                            continue                 
                      else: 
                            F[i]+=mc.e**2/(4*np.pi*mc.epsilon_0)*(x[i]-x[j])/np.linalg.norm(x[i]-x[j])**3
    return F.ravel()
@njit
def mag_field(V,N):
      # B=np.array([0,0,1])
      vel=V.reshape((N,3))
      F=np.zeros((N,3))
      F-=mc.e*(np.cross(vel,B))
      return F.ravel()
"""
random speed generator based on light gas particle at desired temperature with mean and std 
given by the maxwell boltzman distribution
"""
@njit
def rand_v(mu,sigma):
    return np.random.normal(mu,sigma)
#####funcgtion from the internet to generate random unit vectors
    
"""
to generate random unit vectors for collision simulations
"""
@njit
def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    vect=np.zeros(3)
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    #x coord
    vect[0] = np.sin( theta) * np.cos( phi )
    vect[1] = np.sin( theta) * np.sin( phi )
    vect[2] = np.cos( theta )
    return vect
@njit
def rand_n():
    return random_three_vector()

"""
function to calculate new ion velocity after collision with virtual light, 
cold gas particle, based on classical collision
"""
#####v_old must be a numpy vector 3x1. returns a 3x1 numpy array
@njit
def collision(v_old):
    vg=rand_v(mu,sigma)
    n0=random_three_vector() ####random velocity unit vector assigned to ion
    n1=random_three_vector() ####random velocity of gas particle
    return mg/(mc.mb+mg)*np.abs(np.linalg.norm(v_old)-vg)*n0+(mc.mb*v_old+mg*vg*n1)/(mc.mb+mg)
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
def collisions2(V,t):
    X=np.zeros(len(V))
    N=int(len(V)/3)
    if t/(10e-9)-int(t/(10e-9))*10e-9<5e-9:
          for i in range(0,N):
              Y=V[3*i:3*i+3]
              X[3*i:3*i+3]=collision(Y)
          return X
    else: return V


# def excite_motion(vel)
#####funcgtion from the internet to generate random unit vectors
    
"""
to generate random unit vectors for collision simulations
"""


####random position generator
def rand_pos(grid_size):
    return (np.random.random_sample(3)-.5)*grid_size



def initialize_ions(N,start_area,T):
      mu_barium=np.sqrt(8*mc.k*T/(np.pi*mc.mb))
      sigma_barium=np.sqrt(mc.k*T/(2*mc.mb))
      IC=np.zeros(6*N) #initialize output array
      
      for i in range(1,N+1): #loop through ions
            
              IC[3*N+3*(i-1):3*N+3*(i-1)+3] = rand_n() * rand_v(mu_barium,sigma_barium)
              
              IC[3*(i-1):3*(i-1)+3] = rand_pos(start_area) #random position
              
      return IC
def initialize_ions_noz(N,start_area,T):
      mu_barium=np.sqrt(8*mc.k*T/(np.pi*mc.mb))
      sigma_barium=np.sqrt(mc.k*T/(2*mc.mb))
      IC=np.zeros(6*N) #initialize output array
      
      for i in range(1,N+1): #loop through ions
            
              IC[3*N+3*(i-1):3*N+3*(i-1)+3] = rand_n() * rand_v(mu_barium,sigma_barium) *[1,1,0]
              
              IC[3*(i-1):3*(i-1)+3] = rand_pos(start_area)*[1,1,0] #random position
              
      return IC
def initialize_ions3(N,start_area,T):
      return initialize_ions(N,start_area,T).reshape((N,3))
"""
initialize ions and their velocities with random directions and speeds given by temperature desired 
"""

"""
solving the system of ODE's with ode solvers'
"""
#initialize ion trajectories with random displacement and velocity


""" commented out but keeping, in case we want to use ODEint again
@njit
def Newton(X,t,N):
      '''
A few notes about this function
laser cooling and trap functions are 
external since they are simpler and faster. 
      
      
The output is all of the derivatives wrt time, for easy input into a diff eq solver like odeint   
      
      
      '''
      
      dt=np.zeros(6*N)#initialize output array
      
      pos = X[0:3*N]#positions
      vel = X[3*N:6*N]#velocities
      dt[3*N:6*N]=Coulomb_vect(pos,N)/mb + laser_vect(vel,N)/mb + FHYP_vect(pos,N,t)/mb + F_DC(pos,N)/mb
                  #update v_dot (this is F=ma)
      dt[0:3*N] = vel #update x_dot=v
      return dt 
"""



@njit
def Newton3(t,X,N):
      '''
A few notes about this function
laser cooling and trap functions are 
external since they are simpler and faster. 
      
      
The output is all of the derivatives wrt time, for easy input into a diff eq solver like odeint   
      
      
      '''
      mb=mc.mb
      dt=np.zeros(6*N)#initialize output array
      
      pos = X[0:3*N]#positions
      vel = X[3*N:6*N]#velocities
      dt[3*N:6*N]=Coulomb_vect(pos,N)/mb  +F_DC(pos,N)/mb+FHYP_vect(pos,N,t)/mb + laser_vect(vel,N)/mb 
                  #update v_dot (this is F=ma)
                  #######AK you changed this make sure to check it
      dt[0:3*N] = collisions(vel) #vel #update x_dot=v
      return dt


@njit
def Newton5(t,X,N):
      '''
A few notes about this function
laser cooling and trap functions are 
external since they are simpler and faster. 
      
      
The output is all of the derivatives wrt time, for easy input into a diff eq solver like odeint   
      
      
      '''
      mb=mc.mb
      dt=np.zeros(6*N)#initialize output array
      
      pos = X[0:3*N]#positions
      vel = X[3*N:6*N]#velocities
      dt[3*N:6*N]=Coulomb_vect(pos,N)/mb + laser_eq(vel,N,t)/mb + F_DC(pos,N)/mb + FHYP_vect(pos,N,t)/mb 
                  #update v_dot (this is F=ma)
      dt[0:3*N] = vel #update x_dot=v
      # print("hi")
      return dt

"""
use this to simulate trajectories when there is no micromotion. Its much faster and better for reproducing CCD images
"""
###please make T a multiple of tstep
@njit
def leap_frog(N,T,tstep,IC):
      
      #####determine number of iterations
      mb=mc.mb
      iterations=int(T/tstep)
      time_elapsed=np.zeros(iterations)
      # ####initialize trajectories
      trajectories=np.zeros((6*N,iterations),dtype=np.float64)
      trajectories[:,0]=IC
      pos=np.zeros(3*N)
      vel=np.zeros(3*N)
      # trajectories[:,0]=np.array(IC,dtype=np.float64)
      ##initialize  acceleration variable . keep track of old/new
      acc=np.zeros((3*N,2))
      t=0
      pos=trajectories[0:3*N,0].copy()
      vel=trajectories[3*N:6*N,0]
      acc[:,1]=Coulomb_vect(pos,N)/mb  +F_DC(pos,N)/mb+FHYP_pseudo(pos,N)/mb
      it=1
      t=tstep
      
      
      while it<iterations:
            ######get current position,velocity of all ions
            acc[:,0]=acc[:,1].copy()
            pos=trajectories[0:3*N,it-1].copy()
            vel=trajectories[3*N:6*N,it-1].copy()
            ######update positions based on x=vit+1/2at^2
            trajectories[0:3*N,it]=pos+vel*tstep+.5*tstep**2*acc[:,0].copy()
            ########record old acceleration, calculate new one
          
            ######sum up forces
            #with micromotion
            acc[:,1]=1/mb*(Coulomb_vect(pos,N)  +F_DC(pos,N)+FHYP_pseudo(pos,N))# +laser_vect(vel,N))
            #without micromotion
            # acc[:,1]=1/mb*(Coulomb_vect(pos,N)  +F_DC(pos,N)+FHYP_pseudo(pos,N) )

            #####compute velocities
            trajectories[3*N:6*N,it]=vel+.5*tstep*(acc[:,0].copy()+acc[:,1].copy())
            trajectories[3*N:6*N,it]=collisions(trajectories[3*N:6*N,it].copy())
            '''old algo for reference
            # pos_track[it]=pos_track[it-1]+vel_track[it-1]*delta+.5*delta**2*acc_track[it-1]
            # acc_track[it]=sum_force_all(pos_track[it])/mb
            # vel_track[it]=vel_track[it-1]+.5*(acc_track[it]+acc_track[it-1])*delta
            # vel_track[it]=collisions(vel_track[it])
            '''
            time_elapsed[it]=t
            it+=1
            t+=tstep
      return time_elapsed,trajectories

""" Probably phaisng this out
@njit
def Newton4(t,X,N):
      '''
Tested this out as a function to see if it is faster to avoid function calls. Makes it faster but more unreadable
      
      
The output is all of the derivatives wrt time, for easy input into a diff eq solver like odeint   
      
      
      '''
      
      dt=np.zeros(6*N)#initialize output array
      
      pos = X[0:3*N].reshape((N,3))#positions
      vel = X[3*N:6*N].reshape((N,3))#velocities
     
      
 #########     
      F=np.zeros((N,3))
   
      for i in prange(1,N):
            
            perm=pos-np.roll(pos,3*i)
            
            F+=kappa*perm/((np.sqrt((perm**2).sum(axis=1)).repeat(3).reshape((N,3))))**3
      

      ####choose amount to increase/decrease forces by
      
      #calculate force
      F-=mb*trap**2*order*pos
     

#########
      coeff=2*e*V/(r0**2+2*z0**2)*np.cos(omega*t) #calculate the coefficient for the force for a hyperbolic ion trap
      

      ###seperate out the coordinates
      x,y,z=pos[:,0],pos[:,1],pos[:,2]
     
      R,phi=np.sqrt(x**2+y**2),np.arctan2(y,x)    #######convert to polar coordinates
      # phi=np.arctan2(y,x)
      FR=-coeff*R       #calculate the radial trap force

      #restack the array in cartesian coordinates including z force
      
      Fnew=np.stack((FR*np.cos(phi),FR*np.sin(phi),2*coeff*z))
      #return output as flattened array
      F+=Fnew.transpose()


     
#######
      
      
      F0=hbar*K*gamma/2*S/(1+S)
      F+=F0
     
      F=F.ravel()
     
      Beta=-hbar*4*s0*delta/gamma/(1+s0+(2*delta/gamma)**2)**2*np.kron(vel.dot(K),K)
      
      F-=Beta
      
      # F=F.ravel()
      #######
      dt[3*N:6*N]=F/mb
                  #update v_dot (this is F=ma)
      dt[0:3*N] = vel.ravel() #update x_dot=v
      return dt
"""    


# example code for running simulation
# P = INT.solve_ivp(lambda t, y: Newton3(t,y,N), [0,Tfinal],y0=IC, t_eval=t2,method='RK45',max_step=t_int)





