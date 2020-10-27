# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:09:38 2020

@author: Vladimir Sivak
"""

period = 6

# Script of actions
script = {
    'beta'  : [[1.9568148, -1.6200273],  
               [1.3858962, 2.039439 ], 
               [0.80873275, -0.28867772], 
               [-0.04793629, -0.41179013], 
               [-0.0019606 , -0.31774932], 
               [-0.23569003, -0.65272045]],
    'phi' : [[-1.3322645, -3.1327934], 
             [ 1.4414823, -3.141587 ], 
             [-1.1435729,  2.710605 ], 
             [-0.5898812,  3.1374385], 
             [-1.6124026,  3.141461 ], 
             [0.6019088, 3.1083794]]
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'beta'  : [1]*period,
    'phi' : [1]*period
    }