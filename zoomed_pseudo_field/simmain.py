# -*- coding: utf-8 -*-
"""
@author: Alex Kato, Zeyu Ye
@desc An example of using the functions.
@date 9/16/2020
"""

import numpy as np
import fieldfunc as ff
import simfunc as sf
import matplotlib.pyplot as plt

RF=np.array([0,0,0,0,0,0,0,0,1000,1000])
DC=np.array([30,0,0,0,30,0,0,0,0,0])
V0RF, V0DC, cord = ff.trap_potentials(RF, DC)
paramsRF, paramsDC, ncords = ff.curve_fit(V0RF, V0DC, cord)
VRF = ff.fake_V(ncords, paramsRF)
VDC = ff.fake_V(ncords, paramsDC)
Emag = ff.Efield_solve(VRF)
Pseudo = ff.Pseudo_solve(Emag, np.pi*2*12.47*(10**6))
PE = ff.eff_energy(Pseudo, VDC)
F = ff.force_field(PE)
N = 13
IC = sf.initialize_ions(N)
P = sf.leap_frog(N,IC,F)