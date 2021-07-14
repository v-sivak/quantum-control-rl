# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:59:25 2021

@author: qulab
"""
from init_script import *
import numpy as np
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler


class logical_lifetime_with_ECD_init_mixer(FPGAExperiment):
    """
    Experiment to measure GKP logical lifetime.
    
    """
    # Parameters of the stabilization protocol
    params_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000377_sbs_run16.npz')

    # Parameters of the initializing ECDC sequence
    init_params_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000308_gkp_prep_run2.npz')
    init_tau_ns = IntParameter(50)
    
    steps = RangeParameter((1,50,50))
    
    fit_func = {'logical' : 'exp_decay'}
    
    def sequence(self):

        gkp.readout, gkp.qubit, gkp.cavity = readout, qubit, cavity
        
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
            
            # TODO: this mixer update could be done with a subroutine, but 
            # there seem to be some hidden syncs there... 
            phase_reg += float((cavity_phase + np.pi/2.0) / np.pi)
            c = FloatRegister()
            s = FloatRegister()
            c = af_cos(phase_reg)
            s = af_sin(phase_reg)
            DynamicMixer[0][0] <<= c
            DynamicMixer[1][0] <<= s
            DynamicMixer[0][1] <<= -s
            DynamicMixer[1][1] <<= c
            gkp.cavity.delay(gkp.t_mixer_calc_ns)
            gkp.cavity.load_mixer()
            sync()

        sbs_step = gkp.load_sbs_sequence(gkp.s_tau_ns, gkp.b_tau_ns, self.params_filename, version='v3')
        
        ### Parameters of the initialization pulse
        data = np.load(self.init_params_filename, allow_pickle=True)
        beta, phi = data['beta'], data['phi']
        tau = np.array([self.init_tau_ns]*len(data['beta']))
        
        CD_compiler_kwargs = dict(qubit_pulse_pad=gkp.qubit_pulse_pad)
        ECD_control_compiler = ECD_control_simple_compiler(CD_compiler_kwargs, gkp.cal_dir)
        c_pulse, q_pulse = ECD_control_compiler.make_pulse(beta, phi, tau)
        cavity_pulse, qubit_pulse = (c_pulse.real, c_pulse.imag), (q_pulse.real, q_pulse.imag)

        def init_circuit():
            sync()
            gkp.qubit.array_pulse(*qubit_pulse)
            gkp.cavity.array_pulse(*cavity_pulse)
            sync()

        ### Reward measurement circuit 
        @subroutine
        def reward_circuit():
            beta = 1j * np.sqrt(2.0*np.pi) / 2.0
            gkp.reset_feedback_with_echo(gkp.echo_delay, gkp.final_delay, log=True, res_name='m1')
            gkp.displacement_phase_estimation(beta, gkp.t_stabilizer_ns, res_name='m2', amp='dynamic')
            delay(gkp.loop_delay)
        
        ### Experiment
        R = Register()

        with scan_register(*self.steps, reg=R):
            sync()
            gkp.reset_mixer()
            phase_reg = FloatRegister()
            phase_reg <<= 0.0
            sync()
            
            gkp.readout(m0='se')
            init_circuit()
            with Repeat(R):
                sbs_step()
                reset()
                phase_update(phase_reg)
            reward_circuit()


    def process_data(self):
        m0 = self.results['m0'].threshold()
        m1 = self.results['m1'].threshold()
        m2 = self.results['m2'].threshold()
        
        # Account for flipping of the logical Pauli by SBS
        flip_mask = np.array([-1,1,1,-1]*self.steps[-1])[:self.steps[-1]]
        flip_mask[0] *= -1 # need this, becase when R=0 there is no mixer update

#        result = m2.postselect(m0, [0])[0]
        logical_pauli = (1 - 2*m2.thresh_mean().data) * flip_mask
        
        # Every other round the mixer phase is in the wrong qudrature
        ind = np.arange(0, self.steps[-1], 2)
        logical_pauli = logical_pauli[ind]
        rounds = self.results['m0'].ax_data[1][ind]
        
        self.results['logical'] = logical_pauli
        self.results['logical'].ax_data = [rounds]
        self.results['logical'].labels = ['Round']

