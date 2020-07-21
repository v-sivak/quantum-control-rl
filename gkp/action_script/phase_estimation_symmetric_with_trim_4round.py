# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.18
eps = 0.18

period = 4

b_amp = 2*sqrt(pi)
a_amp = sqrt(pi)

# Script of actions
script = {
    'alpha' : [a_amp+0j, -1j*delta, -1j*a_amp, delta+0j],
    'beta'  : [b_amp+0j, eps+0j, 1j*b_amp, 1j*eps],
    'phi' : [pi/2]*4
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [0, 1, 0, 1],
    'beta'  : [0, 1, 0, 1],
    'phi' : [0]*4
    }