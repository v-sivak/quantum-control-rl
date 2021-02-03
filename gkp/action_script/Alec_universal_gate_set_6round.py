# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:09:38 2020

@author: Vladimir Sivak
"""

period = 6

# Script of actions

# # Just empty script
script = {
    'alpha' : [[0.0, 0.0]]*period,
    'beta'  : [[0.0, 0.0]]*period,
    'phi' : [[0.0, 0.0]]*period
    }

# # Corrupted script to create Fock=2
# script = {
#     'alpha' : [[-0.07899787+0.2, -0.4277332],
#                 [ 0.04834698-0.1, -0.18822668],
#                 [0.00700407, 0.13991116+0.26],
#                 [-0.16994776,  0.28379932-0.12],
#                 [0.218085+0.21, 0.27733555],
#                 [0.11619667, 0.05440567]],
#     'beta'  : [[-1.384853+0.1, -1.9636456],
#                 [-1.1933466-0.1,  0.07916671+0.05],
#                 [-0.9131788-0.2,  0.8934469],
#                 [-0.26096117+0.1, -1.2117937 ],
#                 [0.7353108, 0.3276058-0.1],
#                 [0., 0.]],
#     'phi' : [[1.6311785+0.2, 0.57510185],
#               [1.1735618 , 0.93535614-0.1],
#               [ 0.49912393+0.05, -0.37376976],
#               [-0.405657+0.2, -2.58384-0.2],
#               [ 0.39046115+0.11, -1.4893165],
#               [1.2750485 , 0.27563763+0.11]]
#     }


# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'alpha' : [1]*period,
    'beta'  : [1]*(period-1)+[0], # the last round will only have qubit rotations
    'phi' : [1]*period
    }