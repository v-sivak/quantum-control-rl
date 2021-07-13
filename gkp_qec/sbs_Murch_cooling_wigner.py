# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:21:02 2021

@author: qulab
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP


class sbs_Murch_cooling_wigner(FPGAExperiment, GKP):

    disp_range = RangeParameter((-4.0, 4.0, 101))
    reps = IntParameter(5)

    # Baptiste SBS stabilization parameters
    s_tau_ns = IntParameter(10)
    b_tau_ns = IntParameter(150)

    # Murch cooling parameters
    Murch_cool_duration_ns = IntParameter(1000)
    Murch_qubit_ramp_ns = IntParameter(200)
    Murch_readout_ramp_ns = IntParameter(60)
    Murch_readout_amp = FloatParameter(0.22)
    Murch_qubit_amp = FloatParameter(0.2)
    Murch_readout_detune_MHz = FloatParameter(50.4)
    Murch_qubit_detune_MHz = FloatParameter(8.0)
    Murch_qubit_angle = FloatParameter(0.0)
    Murch_qubit_phase = FloatParameter(0.0)

    # misc parameters
    t_mixer_calc = IntParameter(600)
    cavity_phase = FloatParameter(0.0)
    ECD_filename = StringParameter(r'Y:\tmp\for Vlad\from_vlad\000402_sbs_run15.npz')
    echo_delay = IntParameter(868)
    final_delay = IntParameter(64)


    def sequence(self):

        # define mode objects
        self.readout, self.qubit, self.cavity = readout, qubit, cavity_1
        self.qubit_detuned, self.readout_detuned = qubit_ef, readout_aux

        reset = self.reset_autonomous_Murch(self.qubit_detuned, self.readout_detuned,
                self.Murch_cool_duration_ns, self.Murch_qubit_ramp_ns, self.Murch_readout_ramp_ns,
                self.Murch_qubit_amp, self.Murch_readout_amp,
                self.Murch_qubit_detune_MHz, self.Murch_readout_detune_MHz,
                self.Murch_qubit_angle, self.Murch_qubit_phase, 0)

        sbs_step = self.load_sbs_sequence(self.s_tau_ns, self.b_tau_ns, self.ECD_filename, version='v3')

        def phase_update(phase_reg):
            sync()
#            self.qubit_detuned.smoothed_constant_pulse(self.Kerr_drive_time_ns,
#                        amp=self.Kerr_drive_amps[i], sigma_t=self.Kerr_drive_ramp_ns)

            # TODO: this mixer update could be done with a subroutine, but
            # there seem to be some hidden syncs there...
            phase_reg += float((self.cavity_phase + np.pi/2.0) / np.pi)
            c = FloatRegister()
            s = FloatRegister()
            c = af_cos(phase_reg)
            s = af_sin(phase_reg)
            DynamicMixer[0][0] <<= c
            DynamicMixer[1][0] <<= s
            DynamicMixer[0][1] <<= -s
            DynamicMixer[1][1] <<= c
            self.cavity.delay(self.t_mixer_calc)
            self.cavity.load_mixer()
            sync()

        def exp():
            sync()
            phase_reg = FloatRegister()
            phase_reg <<= 0.0
            sync()
            with Repeat(2*self.reps):
                sbs_step()
                reset()
                phase_update(phase_reg)
            sync()


        # map odd parity to 'e'
        with system.wigner_tomography(*self.disp_range, result_name='g_m2',
                                      invert_axis=False):
            readout(g_m0='se')
            exp()
            self.reset_feedback_with_echo(self.echo_delay, self.final_delay,
                                          log=True, res_name='g_m1')

        # map odd parity to 'g'
        with system.wigner_tomography(*self.disp_range, result_name='e_m2',
                                      invert_axis=True):
            readout(e_m0='se')
            exp()
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
