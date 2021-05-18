# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.18

period = 2

b_amp = 2*sqrt(pi)

# Script of actions
script = {
    'alpha' : [[delta ,0], [0,-delta]],
    'beta'  : [[b_amp, 0], [0, b_amp]],
    'phi' : [[pi/2]]*2,
    'theta' : [[0.0]]*2
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [1, 1],
    'beta'  : [0, 0],
    'phi' : [0, 0],
    'theta' : [1, 1]
    }