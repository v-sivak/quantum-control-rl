# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:35:37 2021

@author: Vladimir Sivak
"""
import numpy as np

period = 5

filename = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\fock1.npz'
script_npz = np.load(filename)

script = {} # Script of actions
script['beta'] = [[np.real(b), np.imag(b)] for b in script_npz['betas']]
script['phi'] = [[p,t] for (p,t) in zip(script_npz['phis'], script_npz['thetas'])]


# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'beta'  : [1]*period,
    'phi' : [1]*period
    }