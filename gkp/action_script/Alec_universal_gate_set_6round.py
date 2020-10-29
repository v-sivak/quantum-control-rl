# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:09:38 2020

@author: Vladimir Sivak
"""

period = 6

# Script of actions

# script = {
#     'alpha' : [[0.0, 0.0]]*period,
#     'beta'  : [[0.0, 0.0]]*period,
#     'phi' : [[0.0, 0.0]]*period
#     }

script = {
    'alpha' : [[0.39, 0.27],
              [ 0.16, -0.07],
              [ 0.76, -0.43],
              [0.06, 0.57],
              [-0.11, -0.10],
              [-0.47, -0.04]],
    'beta'  : [[1.69, -0.80],
               [0.42, -1.09],
               [-1.29, -1.84],
               [-1.08, 0.81],
               [0.38, -0.69],
               [1.31, 0.48]],
    'phi' : [[-1.48,  3.13],
             [-0.82,  2.78],
             [-1.04,  2.76],
             [-0.67, -0.43],
             [ 1.08, -1.62],
             [-1.52,  3.12]]
    }

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [1]*period,
    'beta'  : [1]*period,
    'phi' : [1]*period
    }