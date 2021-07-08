# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:48:57 2021

@author: qulab
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP


class sbs_feedback_reset_wigner_1D_Repeat(FPGAExperiment, GKP):

    disp_range = RangeParameter((-4.0, 4.0, 101))
    reps = IntParameter(5)

    # Baptiste SBS stabilization parameters
    s_tau_ns = IntParameter(20)
    b_tau_ns = IntParameter(100)
    ECD_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000402_sbs_run15.npz')

    # Feedback cooling parameters
    echo_delay = IntParameter(0)
    final_delay = IntParameter(0)
    
    phase = FloatParameter(0)


    def sequence(self):

        self.readout, self.qubit, self.cavity = readout, qubit, cavity

        reset = lambda: self.reset_feedback_with_echo(self.echo_delay, self.final_delay)
        sbs_step = self.load_sbs_sequence(self.s_tau_ns, self.b_tau_ns, self.ECD_filename, version='v2')

        def init_circuit():
            pass
        
        def reward_circuit():
            pass

        def control_circuit(s):
            sbs_step(s)
            reset()

        def exp(exp_label):
            sync()

            R0 = Register()
            R1 = Register()
            C = Register()

            R0 <<= 0
            R1 <<= 2*self.reps
            C  <<= 0
            sync()

            label_next('stabilization'+exp_label)

            label_next('x_round'+exp_label)
            if_then(C==R0, 'init'+exp_label)
            if_then(C==R1, 'reward'+exp_label)
            control_circuit('x')
            C += 1
            goto('p_round'+exp_label)

            label_next('p_round'+exp_label)
            if_then(C==R0, 'init'+exp_label)
            if_then(C==R1, 'reward'+exp_label)
            control_circuit('p')
            C += 1
            goto('x_round'+exp_label)

            label_next('init'+exp_label)
            init_circuit()
            C += 1
            goto('stabilization'+exp_label)

            label_next('reward'+exp_label)
            reward_circuit()


        # map odd parity to 'e'
        with system.wigner_tomography_1D(*self.disp_range, result_name='g_m2',
                                      invert_axis=False, phase=self.phase):
            readout(g_m0='se')
            exp(exp_label='g')
            self.reset_feedback_with_echo(self.echo_delay, self.final_delay,
                                          log=True, res_name='g_m1')

        # map odd parity to 'g'
        with system.wigner_tomography_1D(*self.disp_range, result_name='e_m2',
                                      invert_axis=True, phase=self.phase):
            readout(e_m0='se')
            exp(exp_label='e')
            self.reset_feedback_with_echo(self.echo_delay, self.final_delay,
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
