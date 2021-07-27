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

class logical_lifetime_with_ECD_init_mixer_v2(FPGAExperiment):
    """ Experiment to measure GKP logical lifetime. """
    # Parameters of the stabilization protocol
    params_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000377_sbs_run16.npz')

    # Parameters of the initializing ECDC sequence
    Y_init_params_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000308_gkp_prep_run2.npz')
    
    steps = RangeParameter((1,50,50))
    
    fit_func = {'logical_X' : 'exp_decay',
                'logical_Y' : 'exp_decay',
                'logical_Z' : 'exp_decay'}
    
    def sequence(self):

        gkp.readout, gkp.qubit, gkp.cavity, gkp.qubit_detuned = readout, qubit, cavity_1, qubit_ef
        
        ### Parameters of the stabilization protocol
        params = np.load(self.params_filename, allow_pickle=True)
        Kerr_drive_amp = float(params['Kerr_drive_amp'])
        cavity_phases = params['cavity_phase'].squeeze()
        phase_g, phase_e = cavity_phases[0], cavity_phases[1]
        qb_detune, qb_drag = params['detune_reset'], params['drag_reset']
        
        # setup qubit mode for Kerr cancelling drive
        gkp.qubit_detuned.set_detune(gkp.Kerr_drive_detune_MHz*1e6)
        
        def reset(phase_reg, phase_g_reg, phase_e_reg):
            gkp.reset_feedback_with_phase_update(phase_reg, phase_g_reg, phase_e_reg,
                        detune=qb_detune, drag=qb_drag)

        def phase_update(phase_reg):
            sync()
            gkp.qubit_detuned.smoothed_constant_pulse(gkp.Kerr_drive_time_ns,
                        amp=Kerr_drive_amp, sigma_t=gkp.Kerr_drive_ramp_ns)

            gkp.update_phase(phase_reg, gkp.cavity, gkp.t_mixer_calc_ns)
            sync()

        sbs_step = gkp.load_sbs_sequence(gkp.s_tau_ns, gkp.b_tau_ns, self.params_filename, version='v3')
    
        ### Parameters of the initialization pulses 
        cavity_pulse, qubit_pulse = {}, {}

        # Pulse for +Z state
        data = np.load(gkp.plusZ_file, allow_pickle=True)
        cavity_pulse['Z'] = (data['c_pulse'].real, data['c_pulse'].imag)
        qubit_pulse['Z'] = (data['q_pulse'].real, data['q_pulse'].imag)

        # Pulse for +X state
        cavity_pulse['X'] = (-data['c_pulse'].imag, data['c_pulse'].real)
        qubit_pulse['X'] = (data['q_pulse'].real, data['q_pulse'].imag)

        # Pulse for +Y state
        CD_compiler_kwargs = dict(qubit_pulse_pad=gkp.qubit_pulse_pad)
        ECD_control_compiler = ECD_control_simple_compiler(CD_compiler_kwargs, gkp.cal_dir)
        data = np.load(self.Y_init_params_filename, allow_pickle=True)
        beta_Y, phi_Y = data['beta'], data['phi']
        tau = np.array([gkp.init_tau_ns]*len(data['beta']))

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
        
        phase_reg = FloatRegister()
        phase_g_reg = FloatRegister()
        phase_e_reg = FloatRegister()

        phase_g_reg <<= (float(phase_g) + np.pi/2.0) / np.pi
        phase_e_reg <<= (float(phase_e) + np.pi/2.0) / np.pi
        
        
        for s in ['X','Y','Z']:
            with scan_register(*self.steps, reg=R):
                sync()
                gkp.reset_mixer(gkp.cavity, gkp.t_mixer_calc_ns)
                phase_reg <<= 0.0
                sync()
                
                gkp.readout(**{'m0_'+s : 'se'})
                init_circuit(s)
                with Repeat(R):
                    sbs_step()
                    reset(phase_reg, phase_g_reg, phase_e_reg)
                    phase_update(phase_reg)
                    sync()
                reward_circuit(s)

    def process_data(self):        
        
        rounds = np.linspace(*self.steps).astype(int)
        
        # Account for flipping of the logical Pauli by SBS step
        # This mask assumes that protocol starts with 'x' quadrature
        flip_mask = {'X' : np.array([+1,-1,-1,+1]*int(self.steps[1]))[rounds],
                     'Y' : np.array([+1,+1,+1,+1]*int(self.steps[1]))[rounds],
                     'Z' : np.array([+1,+1,-1,-1]*int(self.steps[1]))[rounds]}
        
        # Every other round mixer phase is in the wrong quadrature for X and Z
        # so we need to select only part of the data
        ind = np.where(rounds % 2 == 0)
        
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
