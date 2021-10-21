# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:03:27 2021
"""
from math import sqrt, pi
import numpy as np

period = 3

beta_complex = [0.2j, sqrt(pi/2), 0]
phase, angle = [pi/2, 0, pi/2], [pi/2, -pi/2, pi/2]


script = {} # Script of actions
script['beta'] = [[np.real(b), np.imag(b)] for b in beta_complex]
script['phi'] = [[p,t] for (p,t) in zip(phase, angle)]


mask = {'beta'  : [1]*period, 'phi' : [1]*period}