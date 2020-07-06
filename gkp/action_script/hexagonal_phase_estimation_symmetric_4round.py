# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:46:30 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi, exp

delta = 0.25
eps = 0.25

period = 4

b_amp = sqrt(8*pi/sqrt(3))
a_amp = sqrt(2*pi/sqrt(3))

### Script of actions

# [S_x, trim x, S_z, trim z]
# Trimming and feedback are done in the direction of a different stabilizer,
# instead of the orthogonal direction

beta = [1j*exp(-2j*pi/3)*b_amp, 1j*exp(-2j*pi/3)*eps, 
        1j*b_amp, 1j*eps]

alpha = [1j*exp(-2j*pi/3)*a_amp, -1j*delta, 
         -1j*a_amp, 1j*exp(-2j*pi/3)*delta]

phi = [pi/2, pi/2,
       pi/2, pi/2]

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = [0, 1, 0, 1]