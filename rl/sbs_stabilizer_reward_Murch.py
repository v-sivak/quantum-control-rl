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

__all__ = ['sbs_stabilizer_reward_Murch']

class sbs_stabilizer_reward_Murch(ReinforcementLearningExperiment):
    """ Autonomous GKP stabilization with SBS protocol and Murch cooling."""

    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        self.max_mini_batch_size = 5
        self.batch_axis = 3
        self.s_tau = 20
        self.b_tau = 150
        self.opt_file = r'D:\DATA\exp\2021-06-28_cooldown\sbs_stabilizer_reward\opt_data.npz'

    def update_exp_params(self):

        action_batch = self.message['action_batch']
        self.N_msmt = self.message['N_msmt']
        self.stabilizers = self.message['stabilizers']
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        i_offset = sum(self.mini_batches[:self.mini_batch_idx])
        
        # setup the ECD compiler
        CD_compiler_kwargs = dict(qubit_pulse_pad=GKP().qubit_pulse_pad)
        C = ECD_control_simple_compiler(CD_compiler_kwargs, GKP().cal_dir)
        
        # construct the SBS pulse
        beta, phi = action_batch['beta'][:,0], action_batch['phi'][:,0] # shape=[B,T,2]
        phi_ECD, detune = action_batch['flip'][:,0], action_batch['detune'][:,0]
        tau = np.array([self.s_tau, self.b_tau, self.s_tau, 0])
        self.cavity_pulses, self.qubit_pulses = [], []
        for i in range(mini_batch_size):
            I = i_offset + i
            C_pulse, Q_pulse = C.make_pulse_v2(beta[I], phi[I], phi_ECD[I], tau, detune[I])
            self.cavity_pulses.append(C_pulse)
            self.qubit_pulses.append(Q_pulse)
        logger.info('Constructed waveforms.')
        
        # Murch cooling parameters
        self.Murch_params = {}
        for a in ['Murch_amp', 'Murch_detune_MHz', 'Murch_phi']:
            self.Murch_params[a] = []
            for i in range(mini_batch_size):
                I = i_offset + i
                self.Murch_params[a].append(action_batch[a][I,0])

        # get other experimental parameters
        self.cavity_phases = action_batch['cavity_phase'].squeeze()
        
        # save all parameters to file
        np.savez(self.opt_file, cavity_pulses=self.cavity_pulses, qubit_pulses=self.qubit_pulses,
                 stabilizers=self.stabilizers, cavity_phases=self.cavity_phases,
                 **self.Murch_params)
        
        self.exp = get_experiment(
                'gkp_exp.rl.sbs_stabilizer_reward_Murch_fpga', from_gui=self.use_gui)
        
        self.exp.set_params(
                {'batch_size' : mini_batch_size,
                 'opt_file' : self.opt_file,
                 'n_blocks' : self.N_msmt / 10,
                 'averages_per_block' : 10,
                 'loop_delay' : 4e6,
                 'xp_rounds' : 15,
                 'tau_stabilizer' : 150})

    
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
            
            # Stack the sigma_z results; shape = [N_stabilizers, N_msmt, B]
            sz[m] = np.stack([sz[m][s] for s in self.stabilizers])

        reward_data = np.stack([sz['m1'], sz['m2']]) # shape = [2, N_stabilizers, N_msmt, B]
        R = np.mean(sz['m2'])
        logger.info('Average reward %.3f' %R)
        return reward_data
        