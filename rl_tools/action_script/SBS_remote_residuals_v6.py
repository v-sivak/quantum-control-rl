# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:35:37 2021

@author: Vladimir Sivak
"""
import numpy as np
from math import sqrt, pi

period = 1

# ECD control parameters for SBS pulse
beta_complex = [0.2j, sqrt(2*pi), 0.2j, 0]
alpha_complex = [10, 10, 10, 0]
qb_phase, qb_angle = [pi/2, 0, 0, -pi/2], [pi/2, -pi/2, pi/2, pi/2]

echo_phase, echo_angle = [0, 0, 0], [pi, pi, pi]

cavity_phase = [2.37, -2.41]
Kerr_drive_amp = [0.0, 0.0]
alpha_correction = [0.0, 0.0]
qb_detune = 0.0
qb_drag = 0.0
cavity_detune = 0.0
reset_pulse = [0, pi]

script = {} # Script of actions

script['beta_even'] = [[[np.real(b), np.imag(b)] for b in beta_complex]] # shape=[1,4,2]
script['beta_odd'] = [[[np.real(b), np.imag(b)] for b in beta_complex]] # shape=[1,4,2]
script['phi_even'] = [[[p,a] for (p,a) in zip(qb_phase, qb_angle)]] # shape=[1,4,2]
script['phi_odd'] = [[[p,a] for (p,a) in zip(qb_phase, qb_angle)]] # shape=[1,4,2]
script['echo_pulse_even'] = [[[p,a] for (p,a) in zip(echo_phase, echo_angle)]] # shape=[1,4,2]
script['echo_pulse_odd'] = [[[p,a] for (p,a) in zip(echo_phase, echo_angle)]] # shape=[1,4,2]

script['cavity_phase'] = [cavity_phase] # shape=[1,2]
script['Kerr_drive_amp'] = [Kerr_drive_amp] # shape=[1,2]
script['qb_detune'] = [[qb_detune]] # shape=[1,1]
script['qb_drag'] = [[qb_drag]] # shape=[1,1]
script['cavity_detune'] = [[cavity_detune]] # shape=[1,1]
script['alpha_correction_even'] = [[alpha_correction]*3] # shape=[1,3,2]
script['alpha_correction_odd'] = [[alpha_correction]*3] # shape=[1,3,2]
script['reset_pulse'] = [reset_pulse] # shape=[1,2]

# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {a : [1] for a in script.keys()}