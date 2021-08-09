# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:01:53 2021

@author: qulab
"""
import numpy as np
from rl_client import ReinforcementLearningExperiment
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler
from fpga_lib.scripting import get_experiment

import logging
logger = logging.getLogger('RL')

__all__ = ['state_prep_fock_reward']

class state_prep_fock_reward(ReinforcementLearningExperiment):
    """ State preparation with Fock reward. """

    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        self.max_mini_batch_size = 15
        self.batch_axis = 1
        self.tau_ns = 20
        self.cal_dir = r'D:\DATA\exp\2021-06-28_cooldown\CD_fixed_time_amp_cal'

    def update_exp_params(self):

        action_batch = self.message['action_batch']
        beta, phi = action_batch['beta'], action_batch['phi'] # shape=[B,T,2]
        T = beta.shape[1] # number of time-steps in the sequence
        self.N_msmt = self.message['N_msmt']
        self.fock = self.message['fock']
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        i_offset = sum(self.mini_batches[:self.mini_batch_idx])

        CD_compiler_kwargs = dict(qubit_pulse_pad=4)
        C = ECD_control_simple_compiler(CD_compiler_kwargs, self.cal_dir)
        tau = np.array([self.tau_ns]*T)
        
        self.cavity_pulses, self.qubit_pulses = [], []
        for i in range(mini_batch_size):
            I = i_offset + i
            C_pulse, Q_pulse = C.make_pulse(beta[I], phi[I], tau)
            self.cavity_pulses.append(C_pulse)
            self.qubit_pulses.append(Q_pulse)
        logger.info('Compiled pulses.')
        
        # save pulse sequences to file
        opt_file = r'D:\DATA\exp\2021-04-19_cooldown\state_prep_fock_reward\opt_data.npz'
        np.savez(opt_file, cavity_pulses=self.cavity_pulses, qubit_pulses=self.qubit_pulses)
        
        self.exp = get_experiment(
                'gkp_exp.rl.state_prep_fock_reward_fpga', from_gui=True)
        
        self.exp.set_params(
                {'batch_size' : mini_batch_size,
                 'opt_file' : opt_file,
                 'n_blocks' : self.N_msmt / 2,
                 'averages_per_block' : 2,
                 'fock' : self.fock,
                 'loop_delay' : 4e6})


    def create_reward_data(self):
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        # expected shape of the results is [N_msmt, B]
        m1 = 1. - 2*self.results['m1'].threshold().data
        m2 = 1. - 2*self.results['m2'].threshold().data
        if mini_batch_size == 1:
            m1 = np.expand_dims(m1, 1)
            m2 = np.expand_dims(m2, 1)
        m1 = np.transpose(m1, axes=[1,0])
        m2 = np.transpose(m2, axes=[1,0])
        reward_data = np.stack([m1,m2])
        
        mask = np.where(m1==1., 1., 0.)
        R = - np.mean(m2 * mask)
        logger.info('Average reward %.3f' %R)
        return reward_data
        