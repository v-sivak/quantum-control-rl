# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:35:37 2021

@author: Vladimir Sivak
"""
import numpy as np
from math import sqrt, pi

period = 4

beta_complex = [0.2j, sqrt(2*pi), 0.2j, 0]
q_phase = [pi/2, 0, 0, -pi/2]
q_angle = [pi/2, -pi/2, pi/2, pi/2]

script = {} # Script of actions
script['beta'] = [[np.real(b), np.imag(b)] for b in beta_complex]
script['phi'] = [[p,a] for (p,a) in zip(q_phase, q_angle)]


# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'beta'  : [1]*period,
    'phi' : [1]*period
    }