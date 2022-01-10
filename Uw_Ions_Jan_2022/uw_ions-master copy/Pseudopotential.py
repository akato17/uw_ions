#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:08:58 2019

@author: owner
"""
import numpy as np
import math
####***grid size in metres, size is number of lattice points. V is a NxNxN grid of potential values Feturns NxNxN grid of Efield magnitudes
def Efield_solve(V,size,grid):
    
    Ex=np.zeros((size,size,size))
    Ey=np.zeros((size,size,size))
    Ez=np.zeros((size,size,size))
    Ex,Ey,Ez = np.gradient(V,grid/size)
    Emag=np.zeros((size,size,size))
    for a in range(0,size):
       for b in range(0,size):
          
          for c in range(0,size):
              Emag[a,b,c] = math.sqrt(Ex[a,b,c]**2+Ey[a,b,c]**2+Ez[a,b,c]**2)
    return Emag
###return pseudopotential in eV. E_field must be NxNxN
def Pseudo_solve(E_field,size,grid,omega,M,Q):
    PseudoeV=np.zeros((size,size,size))
    for a in range(0,size):
       for b in range(0,size):
          for c in range(0,size):
              PseudoeV[a,b,c] = Q**2/(4*M*omega**2)*E_field[a,b,c]**2
    return PseudoeV
###tahes dc potential and pseudo potential to give effective potential energy
def Eeff(PS,VDC,Q,size):
    PE=np.zeros((size,size,size),dtype=np.long)
    for a in range(0,size):
       for b in range(0,size):
          for c in range(0,size):
              PE[a,b,c]=Q*VDC[a,b,c]+PS[a,b,c]
    return PE
def Force_field(PE,grid,steps):
    return np.gradient(PE,grid/steps)              