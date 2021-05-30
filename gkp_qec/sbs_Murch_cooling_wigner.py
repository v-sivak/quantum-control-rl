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
    s_tau_ns = IntParameter(20)
    b_tau_ns = IntParameter(50)
    eps1 = FloatParameter(0.2)
    eps2 = FloatParameter(0.2)
    beta = FloatParameter(2.5066) # np.sqrt(2*np.pi)
    s_CD_cal_dir = StringParameter('')
    b_CD_cal_dir = StringParameter('')

    # Murch cooling parameters
    cool_duration_ns = IntParameter(1000)
    qubit_ramp_ns = IntParameter(200)
    readout_ramp_ns = IntParameter(60)
    readout_amp = FloatParameter(0.22)
    qubit_amp = FloatParameter(0.2)
    readout_detune_MHz = FloatParameter(50.4)
    qubit_detune_MHz = FloatParameter(8.0)

    # SNAP parameters
    snap_length = IntParameter(0)


    def sequence(self):

        self.readout = readout
        self.qubit = qubit
        self.cavity = cavity

        reset = self.reset_autonomous_Murch(qubit_ef, readout_aux, self.cool_duration_ns,
                                         self.qubit_ramp_ns, self.readout_ramp_ns,
                                         self.qubit_amp, self.readout_amp,
                                         self.qubit_detune_MHz, self.readout_detune_MHz)

        sbs_step = self.sbs(self.eps1, self.eps2, self.beta,
                            self.s_tau_ns, self.b_tau_ns, self.s_CD_cal_dir, self.b_CD_cal_dir)

        snap = lambda: self.snap(self.snap_length)

        def step(s):
            sbs_step(s)
            reset()
            snap()

        def exp():
            sync()
            with Repeat(self.reps):
                step('x')
                step('p')
            sync()


        with system.wigner_tomography(*self.disp_range, result_name='m2'):
            readout(m0='se')
            exp()
            readout(m1='se')



    def process_data(self):

        # post-select on m0='g'
        # meaning that the whole state prep sequence STARTS from VACUUM
        init_state_g = self.results['m0'].multithresh()[0]
        postselected = self.results['m2'].postselect(init_state_g, [1])[0]
        self.results['m2_postselected_m0'] = postselected
        self.results['m2_postselected_m0'].ax_data = self.results['m2'].ax_data
        self.results['m2_postselected_m0'].labels = self.results['m2'].labels

        self.results['sz_postselected_m0'] = (1 - 2*postselected.threshold())
        self.results['sz_postselected_m0'].ax_data = self.results['m2'].ax_data
        self.results['sz_postselected_m0'].labels = self.results['m2'].labels
        self.results['sz_postselected_m0'].vmin = -1
        self.results['sz_postselected_m0'].vmax = 1


        # post-select on m1='g' outcomes
        # meaning that the whole state prep sequence ENDS in VACUUM
        verification_g = self.results['m1'].multithresh()[0]
        postselected = self.results['m2_postselected_m0'].postselect(verification_g, [1])[0]
        self.results['m2_postselected_m0_m1'] = postselected
        self.results['m2_postselected_m0_m1'].ax_data = self.results['m2'].ax_data
        self.results['m2_postselected_m0_m1'].labels = self.results['m2'].labels

        self.results['sz_postselected_m0_m1'] = (1 - 2*postselected.threshold())
        self.results['sz_postselected_m0_m1'].ax_data = self.results['m2'].ax_data
        self.results['sz_postselected_m0_m1'].labels = self.results['m2'].labels
        self.results['sz_postselected_m0_m1'].vmin = -1
        self.results['sz_postselected_m0_m1'].vmax = 1


        # post-select on m2='g'or'e' outcomes
        # meaning that we eliminate leakage error during tomography
        tomo_not_ge = self.results['m2'].multithresh()[2]
        postselected = self.results['m2_postselected_m0_m1'].postselect(tomo_not_ge, [0])[0]
        self.results['m2_postselected_m0_m1_m2'] = postselected
        self.results['m2_postselected_m0_m1_m2'].ax_data = self.results['m2'].ax_data
        self.results['m2_postselected_m0_m1_m2'].labels = self.results['m2'].labels

        self.results['sz_postselected_m0_m1_m2'] = (1 - 2*postselected.threshold())
        self.results['sz_postselected_m0_m1_m2'].ax_data = self.results['m2'].ax_data
        self.results['sz_postselected_m0_m1_m2'].labels = self.results['m2'].labels
        self.results['sz_postselected_m0_m1_m2'].vmin = -1
        self.results['sz_postselected_m0_m1_m2'].vmax = 1