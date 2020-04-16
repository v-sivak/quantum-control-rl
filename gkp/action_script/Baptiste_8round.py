# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.0
eps = 0.3

period = 8

### Script of actions

beta = [2*sqrt(pi)+0j, 2j*sqrt(pi), 2*sqrt(pi)+0j, 2j*sqrt(pi),
        2*sqrt(pi)+0j, 2j*sqrt(pi), 1*sqrt(pi)+0j, 1j*sqrt(pi)]

alpha = [-1j*delta, delta+0j, -1j*delta, delta+0j, 
         -1j*delta, delta+0j, 0j, 0j]

epsilon = [-1j*eps, eps+0j, -1j*eps, eps+0j,
           -1j*eps, eps+0j, 0j, 0j]

phi = [pi/2, pi/2, pi/2, pi/2, pi/2, pi/2, 0, 0]

### Mask for learned epsilon 

mask = [1, 1, 1, 1, 1, 1, 0, 0]

