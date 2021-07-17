# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:59:25 2021

@author: qulab
"""
from init_script import *
import numpy as np
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler
from fpga_lib.analysis.fit_funcs import exp_decay
import matplotlib.pyplot as plt

class logical_lifetime_with_ECD_init_mixer(FPGAExperiment):
    """ Experiment to measure GKP logical lifetime. """
    # Parameters of the stabilization protocol
    params_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000377_sbs_run16.npz')

    # Parameters of the initializing ECDC sequence
    Z_init_params_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000308_gkp_prep_run2.npz')
    Y_init_params_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000308_gkp_prep_run2.npz')
    init_tau_ns = IntParameter(50)
    
    steps = RangeParameter((1,50,50))
    
    fit_func = {'logical_X' : 'exp_decay',
                'logical_Y' : 'exp_decay',
                'logical_Z' : 'exp_decay'}
    
    def sequence(self):

        gkp.readout, gkp.qubit, gkp.cavity = readout, qubit, cavity_1
        
        ### Parameters of the stabilization protocol
        params = np.load(self.params_filename, allow_pickle=True)
        cavity_phase = float(params['cavity_phase'])
        Kerr_drive_amp = float(params['Kerr_drive_amp'])
        
        # setup qubit mode for Kerr cancelling drive
        self.qubit_detuned = qubit_ef
        self.qubit_detuned.set_detune(gkp.Kerr_drive_detune_MHz*1e6)
        
        reset = lambda: gkp.reset_feedback_with_echo(gkp.echo_delay, 0)

        def phase_update(phase_reg):
            sync()
            self.qubit_detuned.smoothed_constant_pulse(gkp.Kerr_drive_time_ns, 
                        amp=Kerr_drive_amp, sigma_t=gkp.Kerr_drive_ramp_ns)
            
            phase_reg += float((cavity_phase + np.pi/2.0) / np.pi)
            gkp.update_phase(phase_reg, gkp.cavity, gkp.t_mixer_calc_ns)
            sync()

        sbs_step = gkp.load_sbs_sequence(gkp.s_tau_ns, gkp.b_tau_ns, self.params_filename, version='v3')
        
        ### Parameters of the initialization pulses 
        cavity_pulse, qubit_pulse = {}, {}
        
        # Pulse for +Z state
        data = np.load(self.Z_init_params_filename, allow_pickle=True)
        beta_Z, phi_Z = data['beta'], data['phi']
        tau = np.array([self.init_tau_ns]*len(data['beta']))
        
        CD_compiler_kwargs = dict(qubit_pulse_pad=gkp.qubit_pulse_pad)
        ECD_control_compiler = ECD_control_simple_compiler(CD_compiler_kwargs, gkp.cal_dir)
        c_pulse_Z, q_pulse_Z = ECD_control_compiler.make_pulse(beta_Z, phi_Z, tau)
        cavity_pulse['Z'], qubit_pulse['Z'] = (c_pulse_Z.real, c_pulse_Z.imag), (q_pulse_Z.real, q_pulse_Z.imag)

        # Pulse for +X state
        cavity_pulse['X'], qubit_pulse['X'] = (-c_pulse_Z.imag, c_pulse_Z.real), (q_pulse_Z.real, q_pulse_Z.imag)

        # Pulse for +Y state
        data = np.load(self.Y_init_params_filename, allow_pickle=True)
        beta_Y, phi_Y = data['beta'], data['phi']
        tau = np.array([self.init_tau_ns]*len(data['beta']))

        c_pulse_Y, q_pulse_Y = ECD_control_compiler.make_pulse(beta_Y, phi_Y, tau)
        cavity_pulse['Y'], qubit_pulse['Y'] = (c_pulse_Y.real, c_pulse_Y.imag), (q_pulse_Y.real, q_pulse_Y.imag)

        def init_circuit(s):
            sync()
            gkp.qubit.array_pulse(*qubit_pulse[s])
            gkp.cavity.array_pulse(*cavity_pulse[s])
            sync()

        ### Reward measurement circuit 
        @subroutine
        def reward_circuit(s):
            beta = {'X' : np.sqrt(np.pi/2.0),
                    'Y' : (1-1j) * np.sqrt(np.pi/2.0),
                    'Z' : 1j * np.sqrt(np.pi/2.0)}
            gkp.reset_feedback_with_echo(gkp.echo_delay, gkp.final_delay, log=True, res_name='m1_'+s)
            gkp.displacement_phase_estimation(beta[s], gkp.t_stabilizer_ns, res_name='m2_'+s, amp='dynamic')
            delay(gkp.loop_delay)
        
        ### Experiment
        R = Register()
        
        for s in ['X','Y','Z']:
            with scan_register(*self.steps, reg=R):
                sync()
                gkp.reset_mixer(gkp.cavity, gkp.t_mixer_calc_ns)
                phase_reg = FloatRegister()
                phase_reg <<= 0.0
                sync()
                
                gkp.readout(**{'m0_'+s : 'se'})
                init_circuit(s)
                with Repeat(R):
                    sbs_step()
                    reset()
                    phase_update(phase_reg)
                reward_circuit(s)
                
    # TODO: make it compatible with not measuring on T1 after every time-step
    def process_data(self):        
        
        # Account for flipping of the logical Pauli by SBS step
        # This mask assumes that procol starts with 'x' quadrature
        flip_mask = {'X' : np.array([+1,-1,-1,+1]*self.steps[-1])[:self.steps[-1]],
                     'Y' : np.array([+1,-1,+1,-1]*self.steps[-1])[:self.steps[-1]],
                     'Z' : np.array([+1,+1,-1,-1]*self.steps[-1])[:self.steps[-1]]}
        
        # Every other round mixer phase is in the wrong quadrature for X and Z
        # so we need to select only part of the data
        ind = np.arange(0, self.steps[-1], 2)
        
        for s in ['X','Y','Z']:
            logical_pauli = (1 - 2*self.results['m2_'+s].thresh_mean().data) * flip_mask[s]
            logical_pauli = logical_pauli[ind]
            rounds = self.results['m0_'+s].ax_data[1][ind]
            
            self.results['logical_'+s] = logical_pauli
            self.results['logical_'+s].ax_data = [rounds]
            self.results['logical_'+s].labels = ['Round']


    def plot(self, fig, data):
        
        colors = {'X': 'blue',
                  'Y': 'red',
                  'Z': 'green'}
        
        ax = fig.add_subplot(111)
        ax.set_xlabel('Round')
        for s in ['X','Y','Z']:
            rounds = self.results['logical_'+s].ax_data[0]
            ax.plot(rounds, self.results['logical_'+s].data,
                    linestyle='none', marker='.', color=colors[s],
                    label=s+', T=%.0f' % self.fit_params['logical_'+s+':tau'])
            ax.plot(rounds, exp_decay.func(rounds, self.fit_params['logical_'+s+':A'], 
                                           self.fit_params['logical_'+s+':tau'],
                                           self.fit_params['logical_'+s+':ofs']),
                    linestyle='-', color=colors[s])
        ax.grid(True)
        ax.legend()
