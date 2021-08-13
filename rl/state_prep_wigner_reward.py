# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:57:08 2021

@author: qulab
"""
from init_script import *
import numpy as np
import os
from rl_client import ReinforcementLearningExperiment
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler
from fpga_lib.scripting import get_experiment
from fpga_lib import config

import logging
logger = logging.getLogger('RL')

__all__ = ['state_prep_wigner_reward']

class state_prep_wigner_reward(ReinforcementLearningExperiment):
    """ State preparation with Wigner reward. """
    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        self.max_mini_batch_size = 5
        self.batch_axis = 1
        self.tau_ns = 50
        self.qubit_pulse_pad = 4
        self.cal_dir = r'D:\DATA\exp\2021-06-28_cooldown\CD_fixed_time_amp_cal'
        
        self.exp = get_experiment('gkp_exp.rl.state_prep_wigner_reward_fpga', from_gui=self.use_gui)
    
    def update_exp_params(self):

        action_batch = self.message['action_batch']
        beta, phi = action_batch['beta'], action_batch['phi'] # shape=[B,T,2]
        self.alphas = np.array(self.message['mini_buffer'])
        self.targets = np.array(self.message['targets'])
        self.N_alpha = self.message['N_alpha']
        self.N_msmt = self.message['N_msmt']
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        i_offset = sum(self.mini_batches[:self.mini_batch_idx])

        # setup the ECD compiler
        CD_compiler_kwargs = dict(qubit_pulse_pad=self.qubit_pulse_pad)
        C = ECD_control_simple_compiler(CD_compiler_kwargs, self.cal_dir)
        
        # construct the ECD control pulses
        tau = np.array([self.tau_ns] * beta.shape[1])
        self.cavity_pulses, self.qubit_pulses = [], []
        for i in range(mini_batch_size):
            I = i_offset + i
            C_pulse, Q_pulse = C.make_pulse(beta[I], phi[I], tau)
            self.cavity_pulses.append(C_pulse)
            self.qubit_pulses.append(Q_pulse)
        logger.info('Compiled pulses.')

        # save phase space points and pulse sequences to file
        opt_file = os.path.join(config.data_directory, self.exp.name, 'opt_data.npz')
        np.savez(opt_file, alphas=self.alphas, targets=self.targets,
                 cavity_pulses=self.cavity_pulses, qubit_pulses=self.qubit_pulses)

        self.exp.set_params(
                {'batch_size' : mini_batch_size,
                 'N_alpha' : self.N_alpha,
                 'opt_file' : opt_file,
                 'n_blocks' : self.N_msmt / 5,
                 'averages_per_block' : 5,
                 'loop_delay' : 4e6})


    def create_reward_data(self):
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        # expected shape of the results is [N_msmt, B, N_alpha]
        m1 = 1. - 2*self.results['m1'].threshold().data
        m2 = 1. - 2*self.results['m2'].threshold().data
        if mini_batch_size == 1: 
            m1 = np.expand_dims(m1, 1)
            m2 = np.expand_dims(m2, 1)
        m1 = np.transpose(m1, axes=[1,2,0])
        m2 = np.transpose(m2, axes=[1,2,0])
        reward_data = np.stack([m1,m2])
        
        mask = np.where(m1==1., 1., 0.)
        targets = np.reshape(self.targets, [1,len(self.targets),1])
        R = np.mean(m2 * np.sign(targets) * mask)
        logger.info('Average reward %.3f' %R)
        
        return reward_data