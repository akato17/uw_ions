#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:04:07 2019

@author: Alex K.
fits data bear the center of the traqp to get the sewcular frequency
"""

import numpy as np
import spinmob as sm
def fit_secular(Ps,size,grid,M):
    xdat=np.zeros((size))
    ydat=np.zeros((size))
    zdat=np.zeros((size))
    xdat=Ps[:,size/2,size/2][size/2-5:size/2+5]
    ydat=Ps[size/2,:,size/2][size/2-5:size/2+5]
    zdat=Ps[size/2,size/2,:][size/2-5:size/2+5]
    
    space=np.linspace(-grid/2,grid/2,size)[size/2-5:size/2+5]
    fitx=sm.data.fitter()
    fitx.set_functions(f='a*x**2+b*x+c', p='a=1000,b=0,c=0.01')
    fitx.set_data(xdata=space,ydata=xdat)
    fitx.fit()
    wx=np.sqrt(2*fitx.results[0][0]/M)
    fity=sm.data.fitter()
    fity.set_functions(f='a*x**2+b*x+c', p='a=1000,b=0,c=0.01')
    fity.set_data(xdata=space,ydata=ydat)
    fity.fit()
    wy=np.sqrt(2*fity.results[0][0]/M)
    fitz=sm.data.fitter()
    fitz.set_functions(f='a*x**2+b*x+c', p='a=1000,b=0,c=0.01')
    fitz.set_data(xdata=space,ydata=zdat)
    fitz.fit()
    wz=np.sqrt(2*fitz.results[0][0]/M)
    
    return wx,wy,wz