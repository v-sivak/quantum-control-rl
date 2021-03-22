# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:56:21 2021

@author: Vladimir Sivak
"""

period = 8

# Script of actions

# Just empty script
script = {
    'beta'  : [[0.0, 0.0]]*period,
    'phi' : [[0.0, 0.0]]*period
    }


# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'beta'  : [1]*period,
    'phi' : [1]*period
    }