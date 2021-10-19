# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:35:37 2021

@author: Vladimir Sivak
"""
import numpy as np

filename = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\ECDC_sequences\gkp_plusY_T_11_Delta_0.30_F_0.9923.npz'
script_npz = np.load(filename)

period = script_npz['phase'].shape[0]

script = {} # Script of actions
script['beta'] = [[b_re, b_im] for (b_re, b_im) in zip(script_npz['beta_re'], script_npz['beta_im'])]
script['phi'] = [[p,t] for (p,t) in zip(script_npz['phase'], script_npz['angle'])]


# Mask 1 allows the ActionWrapper to use the learned value of the action on
# that time step, while 0 allows to use the scripted value.
mask = {
    'beta'  : [1]*period,
    'phi' : [1]*period
    }