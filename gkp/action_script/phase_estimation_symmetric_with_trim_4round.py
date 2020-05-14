# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.25
eps = 0.25

period = 4

### Script of actions

beta = [2*sqrt(pi)+0j, eps+0j, 2j*sqrt(pi), 1j*eps]

alpha = [sqrt(pi)+0j, -1j*delta, -1j*sqrt(pi), delta+0j]

phi = [pi/2, pi/2, pi/2, pi/2]

mask = [0, 1, 0, 1]