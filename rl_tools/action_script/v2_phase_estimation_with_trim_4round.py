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
    'alpha' : [[a_amp, 0], [0, -delta], [0, -a_amp], [delta, 0]],
    'beta'  : [[b_amp, 0], [eps, 0], [0, b_amp], [0, eps]],
    'phi' : [[pi/2]]*4,
    'theta' : [[0.0]]*4
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [0, 1, 0, 1],
    'beta'  : [0, 1, 0, 1],
    'phi' : [1]*4,
    'theta' : [1]*4
    }