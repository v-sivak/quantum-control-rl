# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:57:08 2021

@author: qulab
"""
from init_script import *
import numpy as np
from rl_client import ReinforcementLearningExperiment
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler
from fpga_lib.scripting import get_experiment

import logging
logger = logging.getLogger('RL')

__all__ = ['state_prep_wigner_reward']

class state_prep_wigner_reward(ReinforcementLearningExperiment):
    """ State preparation with Wigner reward. """

    def update_exp_params(self, message):

        action_batch = message['action_batch']
        beta, phi = action_batch['beta'], action_batch['phi'] # shape=[B,T,2]
        self.alphas = np.array(message['mini_buffer'])
        self.targets = np.array(message['targets'])
        self.N_alpha = message['N_alpha']
        self.N_msmt = message['N_msmt']
        self.batch_size = message['batch_size']
        
        C = ECD_control_simple_compiler(tau_ns=24)
        self.cavity_pulses, self.qubit_pulses = [], []
        for i in range(self.batch_size):
            C_pulse, Q_pulse = C.make_pulse(beta[i], phi[i])
            self.cavity_pulses.append(C_pulse)
            self.qubit_pulses.append(Q_pulse)
        logger.info('Compiled pulses.')
        
        # save phase space points and pulse sequences to file
        opt_file = r'D:\DATA\exp\2021-03-26_cooldown\state_prep_wigner_reward\opt_data.npz'
        np.savez(opt_file, alphas=self.alphas, targets=self.targets,
                 cavity_pulses=self.cavity_pulses, qubit_pulses=self.qubit_pulses)
        
        self.exp = get_experiment(
                'gkp_exp.rl.state_prep_wigner_reward_fpga', from_gui=True)
        
        self.exp.set_params(
                {'batch_size' : self.batch_size,
                 'N_alpha' : self.N_alpha,
                 'opt_file' : opt_file,
                 'n_blocks' : self.N_msmt / 2,
                 'averages_per_block' : 2,
                 'loop_delay' : 4e6})


    def create_reward_data(self, results):
        # expected shape of the results is [N_msmt, B, N_alpha]
        m1 = 1. - 2*results['m1'].threshold().data
        m2 = 1. - 2*results['m2'].threshold().data
        if self.batch_size == 1: 
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
        