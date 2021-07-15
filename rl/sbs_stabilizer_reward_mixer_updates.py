# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:00:57 2021

@author: Vladimir Sivak
"""
import numpy as np
from rl_client import ReinforcementLearningExperiment
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler
from fpga_lib.scripting import get_experiment
from init_script import gkp

import logging
logger = logging.getLogger('RL')

__all__ = ['sbs_stabilizer_reward_mixer_updates']

class sbs_stabilizer_reward_mixer_updates(ReinforcementLearningExperiment):
    """ GKP stabilization with SBS protocol.
    
        In this version the agent learns 
            1) Amplitude of conditional displacements in the ECDC sequence
            2) Parameters of the qubit rotations in the ECDC sequence
            3) Parameters of the echo pi-pulse played during the ECD gates
            4) Detuning of all qubit pulses
            5) Cavity rotation angle
            6) Amplitude of the Kerr-cancelling drive
    """

    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        self.max_mini_batch_size = 5
        self.batch_axis = 3
        self.opt_file = r'D:\DATA\exp\2021-06-28_cooldown\sbs_stabilizer_reward\opt_data.npz'

        self.exp = get_experiment(
                'gkp_exp.rl.sbs_stabilizer_reward_mixer_updates_fpga', from_gui=self.use_gui)

    def update_exp_params(self):

        action_batch = self.message['action_batch']
        self.N_msmt = self.message['N_msmt']
        self.stabilizers = self.message['stabilizers']
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        i_offset = sum(self.mini_batches[:self.mini_batch_idx])
        
        # setup the ECD compiler
        CD_compiler_kwargs = dict(qubit_pulse_pad=gkp.qubit_pulse_pad)
        C = ECD_control_simple_compiler(CD_compiler_kwargs, gkp.cal_dir)
        
        # construct the SBS pulse
        beta, phi = action_batch['beta'][:,0], action_batch['phi'][:,0] # shape=[B,T,2]
        phi_ECD, detune = action_batch['flip'][:,0], action_batch['detune'][:,0]
        tau = np.array([gkp.s_tau_ns, gkp.b_tau_ns, gkp.s_tau_ns, 0])
        self.cavity_pulses, self.qubit_pulses = [], []
        for i in range(mini_batch_size):
            I = i_offset + i
            C_pulse, Q_pulse = C.make_pulse_v2(beta[I], phi[I], phi_ECD[I], tau, detune[I])
            self.cavity_pulses.append(C_pulse)
            self.qubit_pulses.append(Q_pulse)
        logger.info('Constructed waveforms.')
        
        # get other parameters
        self.cavity_phases = action_batch['cavity_phase'].squeeze()
        self.Kerr_drive_amps = action_batch['Kerr_drive_amp'].squeeze()
        
        # save SBS pulse sequences to file
        np.savez(self.opt_file, cavity_pulses=self.cavity_pulses, qubit_pulses=self.qubit_pulses,
                 stabilizers=self.stabilizers, cavity_phases=self.cavity_phases,
                 Kerr_drive_amps=self.Kerr_drive_amps)
        
        self.exp.set_params(
                {'batch_size' : mini_batch_size,
                 'opt_file' : self.opt_file,
                 'n_blocks' : self.N_msmt / 10,
                 'averages_per_block' : 10,
                 'xp_rounds' : 15 + self.epoch % 2})
    

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
        