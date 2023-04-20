# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:18:27 2023

@author: qulab
"""

import numpy as np
from math import sqrt, pi

period = 1


amplitude = 0.3
sigma = 16.0 # arbitrary float
detune = 0.0

script = {} # Script of actions

script['amplitude'] = [[amplitude]] # shape=[1,1]
script['detune'] = [[detune]] # shape=[1,1]

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {a : [1] for a in script.keys()}