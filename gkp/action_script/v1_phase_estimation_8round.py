# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.18
eps = 0.18

period = 8

b_amp = 2*sqrt(pi)
a_amp = 2*sqrt(pi)

# Script of actions
script = {
    'alpha' : [[-a_amp, 0], [0, -delta], 
               [0, -a_amp], [delta, 0], 
               [a_amp, 0], [0, delta], 
               [0, a_amp], [-delta, 0]],
    'beta'  : [[b_amp, 0], [eps, 0], 
               [0, b_amp], [0, eps], 
               [-b_amp, 0], [-eps, 0], 
               [0, -b_amp], [0, -eps]],
    'phi' : [[pi/2]]*8
    }


# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [0,1]*4,
    'beta'  : [0,1]*4,
    'phi' : [1]*8
    }