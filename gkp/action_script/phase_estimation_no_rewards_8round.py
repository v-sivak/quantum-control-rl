# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.2
eps = 0.2

period = 8

### Script of actions

beta = [2*sqrt(pi)+0j, eps+0j, 2j*sqrt(pi), 1j*eps, 
        -2*sqrt(pi)+0j, -eps+0j, -2j*sqrt(pi), -1j*eps]

alpha = [-2*sqrt(pi)+0j, -1j*delta, -2j*sqrt(pi), delta+0j, 
         2*sqrt(pi)+0j, 1j*delta, 2j*sqrt(pi), -delta+0j]

phi = [pi/2, pi/2, pi/2, pi/2, pi/2, pi/2, pi/2, pi/2]
