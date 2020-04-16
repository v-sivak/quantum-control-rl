# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.0
eps = 0.0

period = 4

### Script of actions

beta = [2*sqrt(pi)+0j, 2j*sqrt(pi), 2*sqrt(pi)+0j, 2j*sqrt(pi)]

alpha = [0+0j, 0+0j, -1j*delta, delta+0j]

epsilon = [-1j*eps, eps+0j, -1j*eps, eps+0j]

phi = [0, 0, pi/2, pi/2]







