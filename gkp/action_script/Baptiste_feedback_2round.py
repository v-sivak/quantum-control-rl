# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 22:32:07 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from numpy import sqrt, pi



eps = 0.33

period = 2

### Script of actions

beta = [2*sqrt(pi)+0j, 2j*sqrt(pi)]

alpha = [-1j*sqrt(pi), -sqrt(pi)+0j]

epsilon = [-1j*eps, eps+0j]

mask = [1, 1]