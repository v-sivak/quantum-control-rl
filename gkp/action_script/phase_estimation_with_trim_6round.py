# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.3
eps = 0.3 

period = 6

### Script of actions

beta = [2*sqrt(pi)+0j, eps+0j, 2j*sqrt(pi),
        2j*sqrt(pi), 1j*eps, 2*sqrt(pi)+0j]

alpha = [0+0j, -1j*delta, -1j*sqrt(pi), 
         0+0j, delta+0j, sqrt(pi)+0j, ]

phi = [pi/2, pi/2, 0, pi/2, pi/2, 0]



