# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:00:57 2021

@author: Vladimir Sivak
"""
import numpy as np
from rl_client import ReinforcementLearningExperiment
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler
from fpga_lib.scripting import get_experiment
from gkp_exp.gkp_qec.GKP import GKP

import logging
logger = logging.getLogger('RL')

__all__ = ['sbs_stabilizer_reward']

class sbs_stabilizer_reward(ReinforcementLearningExperiment):
    """ GKP stabilization with SBS protocol. """

    def __init__(self):
        self.max_mini_batch_size = 10
        self.batch_axis = 1
        self.s_tau = 20
        self.b_tau = 50

    def update_exp_params(self):

        action_batch = self.message['action_batch']
        beta, phi = action_batch['beta'], action_batch['phi'] # shape=[B,T,2]
        self.N_msmt = self.message['N_msmt']
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        i_offset = sum(self.mini_batches[:self.mini_batch_idx])
        
        # setup the ECD compiler
        gkp = GKP()
        CD_compiler_kwargs = dict(qubit_pulse_pad=gkp.qubit_pulse_pad)
        cal_dir = r'D:\DATA\exp\2021-05-13_cooldown\CD_fixed_time_amp_cal'
        C = ECD_control_simple_compiler(CD_compiler_kwargs, cal_dir)
        
        # construct the SBS pulse
        tau = np.array([self.s_tau, self.b_tau, self.s_tau, 0])
        self.cavity_pulses, self.qubit_pulses = [], []
        for i in range(mini_batch_size):
            I = i_offset + i
            C_pulse, Q_pulse = C.make_pulse(beta[I], phi[I], tau)
            self.cavity_pulses.append(C_pulse)
            self.qubit_pulses.append(Q_pulse)
        logger.info('Compiled pulses.')
        
        # save SBS pulse sequences to file
        opt_file = r'D:\DATA\exp\2021-05-13_cooldown\sbs_stabilizer_reward\opt_data.npz'
        np.savez(opt_file, cavity_pulses=self.cavity_pulses, qubit_pulses=self.qubit_pulses)
        
        self.exp = get_experiment(
                'gkp_exp.rl.sbs_stabilizer_reward_fpga', from_gui=True)
        
        self.exp.set_params(
                {'batch_size' : mini_batch_size,
                 'opt_file' : opt_file,
                 'n_blocks' : self.N_msmt / 2,
                 'averages_per_block' : 2,
                 'loop_delay' : 5e6,
                 'xp_rounds' : 15,
                 'tau_stabilizer' : 50,
                 'cal_dir' : cal_dir})
    

    def create_reward_data(self):
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        # expected shape of the results is [N_msmt, B]
        m = {}
        for s in ['x','p']:
            m[s] = 1. - 2.*self.results[s].threshold().data
            if mini_batch_size == 1:
                m[s] = np.expand_dims(m[s], 1)
            m[s] = np.transpose(m[s], axes=[1,0])
        
        reward_data = np.stack([m['x'],m['p']])
        R = np.mean(reward_data)
        logger.info('Average reward %.3f' %R)
        return reward_data
        