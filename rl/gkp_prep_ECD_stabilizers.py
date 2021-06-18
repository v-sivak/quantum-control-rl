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

__all__ = ['gkp_prep_ECD_stabilizers']

class gkp_prep_ECD_stabilizers(ReinforcementLearningExperiment):
    """ Learn preparation of the GKP state with ECD control using stabilizer
        values as rewards.    
    """
    def __init__(self):
        self.max_mini_batch_size = 5
        self.batch_axis = 3
        self.tau = 50
        self.opt_file = r'D:\DATA\exp\2021-05-13_cooldown\sbs_stabilizer_reward\opt_data.npz'

    def update_exp_params(self):

        action_batch = self.message['action_batch']
        beta, phi = action_batch['beta'], action_batch['phi'] # shape=[B,T,2]
        self.N_msmt = self.message['N_msmt']
        self.stabilizers = self.message['stabilizers']
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        i_offset = sum(self.mini_batches[:self.mini_batch_idx])
        
        # setup the ECD compiler
        CD_compiler_kwargs = dict(qubit_pulse_pad=GKP().qubit_pulse_pad)
        cal_dir = r'D:\DATA\exp\2021-05-13_cooldown\CD_fixed_time_amp_cal'
        C = ECD_control_simple_compiler(CD_compiler_kwargs, cal_dir)
        
        # construct the ECD control pulses
        tau = np.array([self.tau] * beta.shape[1])
        self.cavity_pulses, self.qubit_pulses = [], []
        for i in range(mini_batch_size):
            I = i_offset + i
            C_pulse, Q_pulse = C.make_pulse(beta[I], phi[I], tau)
            self.cavity_pulses.append(C_pulse)
            self.qubit_pulses.append(Q_pulse)
        logger.info('Constructed waveforms.')
        
        # save pulse sequences to file
        opt_file = r'D:\DATA\exp\2021-05-13_cooldown\sbs_stabilizer_reward\opt_data.npz'
        np.savez(opt_file, cavity_pulses=self.cavity_pulses, qubit_pulses=self.qubit_pulses,
                 stabilizers = self.stabilizers)
        
        self.exp = get_experiment(
                'gkp_exp.rl.gkp_prep_ECD_stabilizers_fpga', from_gui=True)
        
        self.exp.set_params(
                {'batch_size' : mini_batch_size,
                 'opt_file' : opt_file,
                 'n_blocks' : self.N_msmt / 2,
                 'averages_per_block' : 2,
                 'loop_delay' : 5e6,
                 'tau_stabilizer' : 50,
                 'cal_dir' : cal_dir,
                 'stabilizers' : self.stabilizers})
    

#    def create_reward_data(self):
#        mini_batch_size = self.mini_batches[self.mini_batch_idx]
#        # expected shape of the results is [N_msmt, B]
#        m1, m2 = {}, {}
#        stabilizers = np.load(self.opt_file)['stabilizers']
#        for s in stabilizers:
#            m2[s] = 1. - 2.*self.results['m2_'+str(s)].threshold().data
#            m1[s] = 1. - 2.*self.results['m1_'+str(s)].threshold().data
#            if mini_batch_size == 1:
#                m2[s] = np.expand_dims(m2[s], 1)
#                m1[s] = np.expand_dims(m1[s], 1)
#            m2[s] = np.transpose(m2[s], axes=[1,0])
#            m1[s] = np.transpose(m1[s], axes=[1,0])
#        
#        m2_stack = np.stack([m2[s] for s in self.stabilizers])
#        m1_stack = np.stack([m2[s] for s in self.stabilizers])
#        reward_data = np.stack([m1_stack, m2_stack]) # shape = [2, N_stabilizers, B, N_msmt]
#        R = np.mean(reward_data)
#        logger.info('Average reward %.3f' %R)
#        return reward_data


    def create_reward_data(self):
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        stabilizers = np.load(self.opt_file)['stabilizers']
        
        # calculate sigma_z values
        sz = dict(m1={}, m2={})
        for m in ['m1', 'm2']:
            for s in stabilizers:
                # expected shape of the results is [N_msmt, B]
                sz[m][s] = 1. - 2.*self.results[m+'_'+str(s)].threshold().data
                if mini_batch_size == 1:
                    sz[m][s] = np.expand_dims(sz[m][s], 1)
            
            # Stack the stabilizer results; shape = [N_stabilizers, N_msmt, B]
            sz[m] = np.stack([sz[m][s] for s in self.stabilizers])

        reward_data = np.stack([sz['m1'], sz['m2']]) # shape = [2, N_stabilizers, N_msmt, B]
        R = np.mean(sz['m2'])
        logger.info('Average reward %.3f' %R)
        return reward_data