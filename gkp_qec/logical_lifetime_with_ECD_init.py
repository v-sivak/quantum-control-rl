# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:59:25 2021

@author: qulab
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler


class logical_lifetime_with_ECD_init(FPGAExperiment, GKP):

    fit_func = {'logical' : 'exp_decay'}
#    fit_fmt = {'tau' : ('%.2f us', 1e3)}
    
    cal_dir = StringParameter(r'D:\DATA\exp\2021-05-13_cooldown\CD_fixed_time_amp_cal')
    loop_delay = IntParameter(4e6)
    steps = RangeParameter((1,50,50))    

    # Baptiste sbs stabilization parameters
    s_tau_ns = IntParameter(20)
    b_tau_ns = IntParameter(50)
    sbs_ECDC_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000377_sbs_run16.npz')

    # Feedback cooling parameters
    echo_delay = IntParameter(872)
    final_delay = IntParameter(292)

    # Logical initialization parameters
    init_ECDC_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000308_gkp_prep_run2.npz')
    init_tau_ns = IntParameter(50)



    def sequence(self):

        self.readout = readout
        self.qubit = qubit
        self.cavity = cavity
        
        # Parameters of the stabilization protocol
        reset = lambda: self.reset_feedback_with_echo(self.echo_delay, self.final_delay)
        sbs_step = self.load_sbs_sequence(self.s_tau_ns, self.b_tau_ns, self.sbs_ECDC_filename, self.cal_dir, version='v2')

        # Parameters of the initialization pulse
        data = np.load(self.init_ECDC_filename, allow_pickle=True)
        beta, phi = data['beta'], data['phi']
        tau = np.array([self.init_tau_ns]*len(data['beta']))
        
        CD_compiler_kwargs = dict(qubit_pulse_pad=self.qubit_pulse_pad)
        ECD_control_compiler = ECD_control_simple_compiler(CD_compiler_kwargs, self.cal_dir)
        c_pulse, q_pulse = ECD_control_compiler.make_pulse(beta, phi, tau)
        cavity_pulse, qubit_pulse = (c_pulse.real, c_pulse.imag), (q_pulse.real, q_pulse.imag)

        def init_circuit():
            sync()
            qubit.array_pulse(*qubit_pulse)
            cavity.array_pulse(*cavity_pulse)
            sync()


        def control_circuit(s):
            sbs_step(s)
            reset()

        def reward_circuit():
            beta = 1j * np.sqrt(2.0*np.pi) / 2.0
            self.displacement_phase_estimation(beta, self.b_tau_ns, self.cal_dir, res_name='m1')

        R0 = Register(0)
        R1 = Register()
        C = Register()

        with scan_register(*self.steps, reg=R1):
            readout(m0='se')
            sync()
            C <<= 0

            label_next('stabilization')

            label_next('x_round')
            if_then(C==R0, 'init')
            if_then(C==R1, 'reward')
            control_circuit('x')
            C += 1
            goto('p_round')

            label_next('p_round')
            if_then(C==R0, 'init')
            if_then(C==R1, 'reward')
            control_circuit('p')
            C += 1
            goto('x_round')

            label_next('init')
            init_circuit()
            C += 1
            goto('stabilization')

            label_next('reward')
            reward_circuit()
            goto('end')

            label_next('end')
            delay(self.loop_delay)


    def process_data(self):
        m0 = self.results['m0'].threshold()
        m1 = self.results['m1'].threshold()
#        flip_mask = 1 - 2 * (np.linspace(*self.steps) % 2)
        flip_mask = np.array([-1,1,1,-1]*self.steps[-1])[:self.steps[-1]]

        result = m1.postselect(m0, [0])[0]
        logical_pauli = (1-2*result.thresh_mean().data) * flip_mask
        self.results['logical'] = logical_pauli
        self.results['logical'].ax_data = self.results['m0'].ax_data[1:]
        self.results['logical'].labels = self.results['m0'].labels[1:]

#
#    def plot(self, fig, data):
#
#        ax = fig.add_subplot(111)
#        ax.set_xlabel('Number of x/p rounds')
#        ax.set_ylabel('sigma_z')
#        for s in ['x','p']:
#            ax.plot(self.results[s+'_sigma_z'].ax_data[1],
#                    self.results[s+'_sigma_z'].data.mean(axis=0),
#                    marker='.')
