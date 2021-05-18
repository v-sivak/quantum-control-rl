# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:49:57 2020

@author: Vladimir Sivak
"""

period = 1

### Script of actions
script = {
    'alpha' : [[5.]],
    'phi_g' : [[0.]],
    'phi_e' : [[0.]]
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [1],
    'phi_e' : [1],
    'phi_g' : [1]
    }
