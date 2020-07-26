# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:49:26 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi, exp

delta = 0.19
eps = 0.19

period = 12

b_amp = sqrt(8*pi/sqrt(3))
a_amp = sqrt(2*pi/sqrt(3))

### Script of actions

script = {
    'alpha' : [1j*a_amp*exp(-0j*pi/3), 
                delta*exp(-2j*pi/3), delta*exp(-1j*pi/3), delta*exp(-0j*pi/3), 
                1j*a_amp*exp(-2j*pi/3),
                delta*exp(-2j*pi/3), delta*exp(-1j*pi/3), delta*exp(-0j*pi/3),
                1j*a_amp*exp(-1j*pi/3),
                delta*exp(-2j*pi/3), delta*exp(-1j*pi/3), delta*exp(-0j*pi/3)], 
      
    'beta'  : [1j*b_amp*exp(-2j*pi/3), 1j*b_amp*exp(-1j*pi/3), 1j*b_amp*exp(-0j*pi/3),
                -eps*exp(-2j*pi/3),
                1j*b_amp*exp(-2j*pi/3), 1j*b_amp*exp(-1j*pi/3), 1j*b_amp*exp(-0j*pi/3),
                -eps*exp(-1j*pi/3),
                1j*b_amp*exp(-2j*pi/3), 1j*b_amp*exp(-1j*pi/3), 1j*b_amp*exp(-0j*pi/3),
                -eps*exp(-0j*pi/3)],
    
    'phi' : [pi/2]*12
    }
    

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [0, 1, 1, 1]*3,
    'beta'  : [0, 0, 0, 1]*3,
    'phi'   : [0]*12
    }