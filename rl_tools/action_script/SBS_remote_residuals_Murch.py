# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:35:37 2021

@author: Vladimir Sivak
"""
import numpy as np
from math import sqrt, pi

period = 1
script = {} # Script of actions

# ECD control parameters for SBS pulse
beta_complex = [0.2j, sqrt(2*pi), 0.2j, 0]
qb_phase, qb_angle = [pi/2, 0, 0, -pi/2], [pi/2, -pi/2, pi/2, pi/2]
flip_phase, flip_angle = [0, 0, 0, 0], [pi, pi, pi, pi]
detune_flip, detune_ecd = [0, 0, 0, 0], [0, 0, 0, 0]

# Murch cooling parameters
qubit_amp = 0.6
readout_amp = 0.645
qubit_detune_MHz = 15.85
readout_detune_MHz = 56
phase, angle = 0, 0


# Create the script of actions. The fist dummy dimension is the time-step dimension

script['beta'] = [[[np.real(b), np.imag(b)] for b in beta_complex]] # shape=[1,4,2]
script['phi'] = [[[p,a] for (p,a) in zip(qb_phase, qb_angle)]] # shape=[1,4,2]
script['flip'] = [[[p,a] for (p,a) in zip(flip_phase, flip_angle)]] # shape=[1,4,2]
script['detune'] = [[[p,a] for (p,a) in zip(detune_flip, detune_ecd)]] # shape=[1,4,2]

script['Murch_amp'] = [[qubit_amp, readout_amp]] # shape=[1,2]
script['Murch_detune_MHz'] = [[qubit_detune_MHz, readout_detune_MHz]] # shape=[1,2]
script['Murch_phi'] = [[phase, angle]] # shape=[1,2]

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {a : [1] for a in script.keys()}