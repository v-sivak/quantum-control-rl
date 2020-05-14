# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:57:21 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
from numpy import sqrt, pi

delta = 0.2

period = 8

### Script of actions

beta = [2*sqrt(pi)+0j] * 4 + [2j*sqrt(pi)] * 4

alpha = [delta+0j, -1j*delta, -1j*delta, -1j*delta,
         -1j*delta, delta+0j, delta+0j, delta+0j]

phi = [pi/2] * 8