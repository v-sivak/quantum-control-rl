# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:03:29 2021

@author: qulab
"""
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler
from init_script import *


class gkp_state_prep_wigner(FPGAExperiment):
    """
    Load ECD control sequence from file and take the Wigner of the state.

    """
    disp_range = RangeParameter((-4.0, 4.0, 81))
    tau_ns = IntParameter(24)
    filename = StringParameter(r'C:\code\gkp_exp\state_prep\fock4.npz')

    def sequence(self):
        data = np.load(self.filename, allow_pickle=True)
        beta, phi = data['beta'], data['phi']
        tau = np.array([self.tau_ns]*len(data['beta']))

        gkp.readout, gkp.qubit, gkp.cavity = readout, qubit, cavity

        # setup the ECD compiler
        CD_compiler_kwargs = dict(qubit_pulse_pad=gkp.qubit_pulse_pad)
        C = ECD_control_simple_compiler(CD_compiler_kwargs, gkp.cal_dir)
        self.c_pulse, self.q_pulse = C.make_pulse(beta, phi, tau)

        def ECDC_sequence():
            sync()
            gkp.qubit.array_pulse(self.q_pulse.real, self.q_pulse.imag)
            gkp.cavity.array_pulse(self.c_pulse.real, self.c_pulse.imag)
            sync()

        # map odd parity to 'e'
        with system.wigner_tomography(*self.disp_range, result_name='g_m2',
                                      invert_axis=False):
            readout(g_m0='se')
            ECDC_sequence()
            gkp.reset_feedback_with_echo(gkp.echo_delay, gkp.final_delay,
                                          log=True, res_name='g_m1')

        # map odd parity to 'g'
        with system.wigner_tomography(*self.disp_range, result_name='e_m2',
                                      invert_axis=True):
            readout(e_m0='se')
            ECDC_sequence()
            gkp.reset_feedback_with_echo(gkp.echo_delay, gkp.final_delay,
                                          log=True, res_name='e_m1')


    def process_data(self):
        sign = dict(g=1, e=-1)

        for s in ['g', 'e']:

            # post-select on m0='g'
            # meaning that the whole state prep sequence STARTS from VACUUM
            init_state_g = self.results[s+'_m0'].multithresh()[0]
            postselected = self.results[s+'_m2'].postselect(init_state_g, [1])[0]
            self.results[s+'_m2_postselected_m0'] = postselected
            self.results[s+'_m2_postselected_m0'].ax_data = self.results[s+'_m2'].ax_data
            self.results[s+'_m2_postselected_m0'].labels = self.results[s+'_m2'].labels

            self.results[s+'_sz_postselected_m0'] = sign[s]*(1 - 2*postselected.threshold())
            self.results[s+'_sz_postselected_m0'].ax_data = self.results[s+'_m2'].ax_data
            self.results[s+'_sz_postselected_m0'].labels = self.results[s+'_m2'].labels
            self.results[s+'_sz_postselected_m0'].vmin = -1
            self.results[s+'_sz_postselected_m0'].vmax = 1


            # post-select on m1='g' outcomes
            # meaning that the whole state prep sequence ENDS in VACUUM
            verification_g = self.results[s+'_m1'].multithresh()[0]
            postselected = self.results[s+'_m2_postselected_m0'].postselect(verification_g, [1])[0]
            self.results[s+'_m2_postselected_m0_m1'] = postselected
            self.results[s+'_m2_postselected_m0_m1'].ax_data = self.results[s+'_m2'].ax_data
            self.results[s+'_m2_postselected_m0_m1'].labels = self.results[s+'_m2'].labels

            self.results[s+'_sz_postselected_m0_m1'] = sign[s]*(1 - 2*postselected.threshold())
            self.results[s+'_sz_postselected_m0_m1'].ax_data = self.results[s+'_m2'].ax_data
            self.results[s+'_sz_postselected_m0_m1'].labels = self.results[s+'_m2'].labels
            self.results[s+'_sz_postselected_m0_m1'].vmin = -1
            self.results[s+'_sz_postselected_m0_m1'].vmax = 1


            # post-select on m2='g'or'e' outcomes
            # meaning that we eliminate leakage error during tomography
            tomo_not_ge = self.results[s+'_m2'].multithresh()[2]
            postselected = self.results[s+'_m2_postselected_m0_m1'].postselect(tomo_not_ge, [0])[0]
            self.results[s+'_m2_postselected_m0_m1_m2'] = postselected
            self.results[s+'_m2_postselected_m0_m1_m2'].ax_data = self.results[s+'_m2'].ax_data
            self.results[s+'_m2_postselected_m0_m1_m2'].labels = self.results[s+'_m2'].labels

            self.results[s+'_sz_postselected_m0_m1_m2'] = sign[s]*(1 - 2*postselected.threshold())
            self.results[s+'_sz_postselected_m0_m1_m2'].ax_data = self.results[s+'_m2'].ax_data
            self.results[s+'_sz_postselected_m0_m1_m2'].labels = self.results[s+'_m2'].labels
            self.results[s+'_sz_postselected_m0_m1_m2'].vmin = -1
            self.results[s+'_sz_postselected_m0_m1_m2'].vmax = 1

        # average results of 'g' and 'e' maps to cancel background
        avg_sz =  0.5 * (self.results['g_sz_postselected_m0'].data.mean(axis=0) + self.results['e_sz_postselected_m0'].data.mean(axis=0))
        self.results['avg_sz_postselected_m0'] = avg_sz
        self.results['avg_sz_postselected_m0'].ax_data = self.results['g_m2'].ax_data[1:]
        self.results['avg_sz_postselected_m0'].labels = self.results['g_m2'].labels[1:]
        self.results['avg_sz_postselected_m0'].vmin = -1
        self.results['avg_sz_postselected_m0'].vmax = 1

        avg_sz =  0.5 * (self.results['g_sz_postselected_m0_m1'].data.mean(axis=0) + self.results['e_sz_postselected_m0_m1'].data.mean(axis=0))
        self.results['avg_sz_postselected_m0_m1'] = avg_sz
        self.results['avg_sz_postselected_m0_m1'].ax_data = self.results['g_m2'].ax_data[1:]
        self.results['avg_sz_postselected_m0_m1'].labels = self.results['g_m2'].labels[1:]
        self.results['avg_sz_postselected_m0_m1'].vmin = -1
        self.results['avg_sz_postselected_m0_m1'].vmax = 1

        avg_sz =  0.5 * (self.results['g_sz_postselected_m0_m1_m2'].data.mean(axis=0) + self.results['e_sz_postselected_m0_m1_m2'].data.mean(axis=0))
        self.results['avg_sz_postselected_m0_m1_m2'] = avg_sz
        self.results['avg_sz_postselected_m0_m1_m2'].ax_data = self.results['g_m2'].ax_data[1:]
        self.results['avg_sz_postselected_m0_m1_m2'].labels = self.results['g_m2'].labels[1:]
        self.results['avg_sz_postselected_m0_m1_m2'].vmin = -1
        self.results['avg_sz_postselected_m0_m1_m2'].vmax = 1
