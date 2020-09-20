# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:56:05 2020

@author: Vladimir Sival


This protocol repeatedly does phase estimation on the stabilizers of the X+ 
state. These is no trimming. It should be used with 'v1' quantum circuit.

"""

import tensorflow as tf
from numpy import sqrt, pi

delta = 0.22

period = 4

b_amp = 2*sqrt(pi)

# Script of actions
script = {
    'alpha' : [-delta+0j, -1j*delta, delta+0j, 1j*delta],
    'beta'  : [b_amp/2+0j, 1j*b_amp, -b_amp/2+0j, -1j*b_amp],
    'phi' : [pi/2]*4,
    'theta' : [0.0]*4
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [1]*4,
    'beta'  : [0]*4,
    'phi' : [1]*4,
    'theta' : [1]*4
    }