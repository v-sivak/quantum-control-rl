# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 22:32:07 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



eps = 0.33

period = 2

b_amp = 2*sqrt(pi)
a_amp = sqrt(pi)

# Script of actions
script = {
    'alpha' : [[0, -a_amp], [-a_amp, 0]],
    'beta'  : [[b_amp, 0], [0, b_amp]],
    'epsilon' : [[0, -eps], [eps, 0]]
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [0, 0],
    'beta'  : [0,0],
    'epsilon' : [1, 1]
    }