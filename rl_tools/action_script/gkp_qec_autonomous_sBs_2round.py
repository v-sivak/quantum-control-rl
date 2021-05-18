# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:47:17 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi

delta = 0.0
eps = 0.20

period = 2

b_amp = 2*sqrt(pi)

### Script of actions
script = {
    'beta'  : [[b_amp, 0], [0, b_amp]],
    'eps1' : [[0, eps], [-eps, 0]],
    'eps2' : [[0, eps], [-eps, 0]],
    'phi' : [[0]]*2,
    'theta' : [[0]]*2 # -0.01
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'beta'  : [1, 1],
    'eps1' : [1, 1],
    'eps2' : [1, 1],
    'phi' : [1, 1],
    'theta' : [1, 1]
    }
