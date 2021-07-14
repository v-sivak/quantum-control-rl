# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:48:57 2021

@author: qulab
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:21:02 2021

@author: qulab
"""
from init_script import *
import numpy as np


class sbs_feedback_reset_wigner_Mixer(FPGAExperiment):

    disp_range = RangeParameter((-4.0, 4.0, 101))
    xp_rounds = IntParameter(15)
    params_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000402_sbs_run15.npz')

    def sequence(self):

        gkp.readout, gkp.qubit, gkp.cavity = readout, qubit, cavity_1
        
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

        def exp():
            sync()
            gkp.reset_mixer()
            phase_reg = FloatRegister()
            phase_reg <<= 0.0
            sync()
            with Repeat(2*self.xp_rounds):
                sbs_step()
                reset()
                phase_update(phase_reg)
            sync()

        # map odd parity to 'e'
        with system.wigner_tomography(*self.disp_range, result_name='g_m2',
                                      invert_axis=False):
            gkp.readout(g_m0='se')
            exp()
            gkp.reset_feedback_with_echo(gkp.echo_delay, gkp.final_delay,
                                          log=True, res_name='g_m1')

        # map odd parity to 'g'
        with system.wigner_tomography(*self.disp_range, result_name='e_m2',
                                      invert_axis=True):
            gkp.readout(e_m0='se')
            exp()
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