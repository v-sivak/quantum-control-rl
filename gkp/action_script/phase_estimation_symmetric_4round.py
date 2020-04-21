# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:27:08 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



delta = 0.2


period = 4

### Script of actions

beta = [2*sqrt(pi)+0j, 2j*sqrt(pi), 2*sqrt(pi)+0j, 2j*sqrt(pi)]

alpha = [delta+0j, 0+0j, 0+0j, -1j*delta]

phi = [0, 0, pi/2, pi/2]