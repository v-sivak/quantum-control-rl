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

from fpga_lib.scripting import wait_complete, get_last_results

import logging
logger = logging.getLogger('RL')

__all__ = ['sbs_Pauli_reward']

class sbs_Pauli_reward(ReinforcementLearningExperiment):
    """ GKP stabilization with SBS protocol.
    
        In this version the agent learns 
            1) Amplitude of conditional displacements in SBS
            2) Parameters of the qubit rotations in SBS
            3) Parameters of the echo pi-pulse played during the CD gates
            4) Detuning & drag all qubit pulses in SBS
            5) Cavity rotation angle for 'g' and 'e' outcomes
            6) Amplitude of the Kerr-cancelling drive
            7) Corrections to large displacement 'alpha' in the CD gate
            8) Detuning & drag of the feedback pulse in the reset
    
        Reward is logical Pauli values of the ideal code.
    """

    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        self.max_mini_batch_size = 10
        self.batch_axis = 3
        self.opt_file = r'D:\DATA\exp\2021-06-28_cooldown\sbs_stabilizer_reward\opt_data.npz'

        self.exp = get_experiment('gkp_exp.rl.sbs_Pauli_reward_fpga_v2', from_gui=self.use_gui)

    
    def rounds_schedule(self, epoch):
        if epoch < 50:
            return 4
        if epoch < 200:
            return 16
        return 48

    
    def update_exp_params(self):

        action_batch = self.message['action_batch']
        self.N_msmt = self.message['N_msmt']
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        i_offset = sum(self.mini_batches[:self.mini_batch_idx])
        
        # setup the ECD compiler
        CD_compiler_kwargs = dict(qubit_pulse_pad=gkp.qubit_pulse_pad)
        C = ECD_control_simple_compiler(CD_compiler_kwargs, gkp.cal_dir)
        
        # construct the SBS pulse
        beta, phi = action_batch['beta'][:,0], action_batch['phi'][:,0] # shape=[B,T,2]
        phi_ECD, alpha_correction = action_batch['flip'][:,0], action_batch['alpha_correction'][:,0]
        self.qb_drag = action_batch['qb_drag'].squeeze()
        self.qb_detune = action_batch['qb_detune'].squeeze()

        tau = np.array([gkp.s_tau_ns, gkp.b_tau_ns, gkp.s_tau_ns, 0])
        self.cavity_pulses, self.qubit_pulses = [], []
        for i in range(mini_batch_size):
            I = i_offset + i
            detune, drag = self.qb_detune[I]*np.ones([4,2]), self.qb_drag[I]*np.ones([4,2])
            C_pulse, Q_pulse = C.make_pulse_v2(beta[I], phi[I], phi_ECD[I], tau, 
                                               detune, alpha_correction[I], drag)
            self.cavity_pulses.append(C_pulse)
            self.qubit_pulses.append(Q_pulse)
        
        # get other parameters
        self.cavity_phases = action_batch['cavity_phase'].squeeze()
        self.Kerr_drive_amps = action_batch['Kerr_drive_amp'].squeeze()

        # Construct the X initialization pulse        
        data = np.load(gkp.plusZ_file, allow_pickle=True)
        self.init_cavity_pulse, self.init_qubit_pulse = data['c_pulse'], data['q_pulse']
        self.init_cavity_pulse *= 1j # need a pulse for X state, not Z state
        logger.info('Constructed waveforms.')

        # save experimental parameters to file
        np.savez(self.opt_file, cavity_pulses=self.cavity_pulses, qubit_pulses=self.qubit_pulses,
                 cavity_phases=self.cavity_phases, Kerr_drive_amps=self.Kerr_drive_amps,
                 init_cavity_pulse=self.init_cavity_pulse, init_qubit_pulse=self.init_qubit_pulse,
                 qb_drag=self.qb_drag, qb_detune=self.qb_detune)
        
        # rounds has to be multiple of 4
        self.exp.set_params(
                {'batch_size' : mini_batch_size,
                 'opt_file' : self.opt_file,
                 'n_blocks' : self.N_msmt / 10,
                 'averages_per_block' : 10,
                 'rounds' : self.rounds_schedule(self.epoch)})


    def create_reward_data(self):
        mini_batch_size = self.mini_batches[self.mini_batch_idx]
        paulis = ['X', 'Z']
        
        # calculate sigma_z values
        sz = dict(m1={}, m2={})
        for m in ['m1', 'm2']:
            for s in paulis:
                # expected shape of the results is [N_msmt, B]
                sz[m][s] = 1. - 2.*self.results[m+'_'+s].threshold().data
                if mini_batch_size == 1:
                    sz[m][s] = np.expand_dims(sz[m][s], 1)
            
            # Stack the sigma_z results; shape = [N_paulis, N_msmt, B]
            sz[m] = np.stack([sz[m][s] for s in paulis])

        reward_data = np.stack([sz['m1'], sz['m2']]) # shape = [2, N_paulis, N_msmt, B]
        R = np.mean(sz['m2'])
        logger.info('Average reward %.3f' %R)
        
        if R > 0.8:
            logger.info('FPGA error, re-collecting mini-batch %d.' %self.mini_batch_idx)
            self.exp.run()
            wait_complete(self.exp)
            logger.info('Finished collecting mini-batch %d.' %self.mini_batch_idx)
            self.results = get_last_results(self.exp)
            return self.create_reward_data()
        
        return reward_data
        