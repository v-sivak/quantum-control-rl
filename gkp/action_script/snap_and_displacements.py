# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:27:55 2020

@author: Vladimir Sivak
"""

period = 5
snap_levels = 7 #14

# Script of actions

script = {
    'alpha' : [[0.0, 0.0]]*period,
    'theta'  : [[0.0]*snap_levels]*period
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [1]*period,
    'theta'  : [1]*period
    }