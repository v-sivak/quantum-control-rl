# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:09:38 2020

@author: Vladimir Sivak
"""

period = 6

# Script of actions
script = {
    'alpha' : [[0.0, 0.0]]*period,
    'beta'  : [[0.0, 0.0]]*period,
    'phi' : [[0.0, 0.0]]*period
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [1]*period,
    'beta'  : [1]*period,
    'phi' : [1]*period
    }