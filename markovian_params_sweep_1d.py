# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:13:46 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from numpy import sqrt, pi, exp
from scipy.optimize import curve_fit

from rl_tools.tf_env import helper_functions as hf
from rl_tools.tf_env import env_init
from rl_tools.tf_env import policy as plc


class ActionScript(object):
    
    def __init__(self, param):
        
        delta = param
        self.period = 2
        b_amp = 2*sqrt(pi)
        a_amp = sqrt(pi)
        
        self.script = {
            'alpha' : [delta+0j, -1j*delta],
            'beta'  : [b_amp+0j, 1j*b_amp],
            'phi' : [pi/2]*2,
            'theta' : [0.0]*2
            }


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

env = env_init(control_circuit='oscillator', init='Z+', H=1, batch_size=1000, 
               episode_length=200, reward_mode = 'fidelity', channel='diffusion',
               quantum_circuit_type='v2', encoding = 'square', N=200)

savepath = r'E:\VladGoogleDrive\Qulab\GKP\sims\diffusion_channel'
params = [0j] + list(np.linspace(0.0, 0.5, 11, dtype=complex))
make_figure = True

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

lifetimes = np.zeros(len(params))

for j in range(len(params)):
    t = time()
    action_script = ActionScript(param=params[j])
    policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)
        
    popt = hf.fit_logical_lifetime(env, policy, plot=False, save_dir=None, 
                 states=['Z+'], reps=1)
    lifetimes[j] = popt['Z+'][1]*1e6
    
    print('(%d): Time %.3f sec' %(j, time()-t))


# Plot summary of the sweep and save the sweep data
if make_figure:
    fig, ax = plt.subplots(1,1, dpi=300, figsize=(10,6))
    ax.set_title(r'Logical lifetime')
    ax.set_ylabel(r'$T_Z$')
    ax.set_xlabel('Sweep parameter')
    ax.plot(params, lifetimes)
    
    fig.savefig(os.path.join(savepath,'summary.png'))

np.save(os.path.join(savepath,'params'), params)
np.save(os.path.join(savepath,'lifetimes'), lifetimes)