# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:06:26 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.18
eps = 0.18

period = 8

b_amp = 2*sqrt(pi)
a_amp = sqrt(pi)

# Script of actions
script = {
    'alpha' : [a_amp+0j] + [-1j*delta, delta+0j]*3 + [-1j*a_amp],
    'beta' : [b_amp+0j, 1j*b_amp]*3 + [eps+0j, 1j*eps],
    'phi' : [pi/2]*period,
    'theta' : [0.0]*period
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [0] + [1, 1]*3 + [0],
    'beta' : [0, 0]*3 + [1, 1],
    'phi' : [1]*period,
    'theta': [1]*period
    }
