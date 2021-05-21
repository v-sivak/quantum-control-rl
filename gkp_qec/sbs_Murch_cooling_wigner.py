# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:21:02 2021

@author: qulab
"""
from init_script import *
from gkp_exp.CD_gate.conditional_displacement_compiler import SBS_simple_compiler
import numpy as np

class sbs_Murch_cooling_wigner(FPGAExperiment):

    disp_range = RangeParameter((-4.0, 4.0, 101))
    reps = IntParameter(5)

    # Baptiste sbs stabilization parameters
    tau_ns = IntParameter(36)
    eps1 = FloatParameter(0.2)
    eps2 = FloatParameter(0.2)
    beta = FloatParameter(2.5066) # np.sqrt(2*np.pi)
    cal_dir = StringParameter('')

    # Murch cooling parameters
    cool_duration_ns = IntParameter(1000)
    qubit_ramp_ns = IntParameter(200)
    readout_ramp_ns = IntParameter(60)
    readout_amp = FloatParameter(0.22)
    qubit_amp = FloatParameter(0.2)
    readout_detune_MHz = FloatParameter(50.4)
    qubit_detune_MHz = FloatParameter(8.0)



    def sequence(self):

        self.readout = readout
        self.qubit = qubit
        self.cavity = cavity
        
        # setup Murch cooling ------------------------------------------------
        # --------------------------------------------------------------------
        self.qubit_detuned = qubit_ef
        self.readout_detuned = readout_aux

        sync()
        self.readout_detuned.set_detune(self.readout_detune_MHz*1e6)
        self.qubit_detuned.set_detune(self.qubit_detune_MHz*1e6)
        sync()

        readout_pump_time = self.cool_duration_ns+2*self.qubit_ramp_ns-2*self.readout_ramp_ns

        def cooling():
            sync()
            self.readout_detuned.smoothed_constant_pulse(
                    readout_pump_time, amp=self.readout_amp, sigma_t=self.readout_ramp_ns)
            self.qubit_detuned.smoothed_constant_pulse(
                    self.cool_duration_ns, amp=self.qubit_amp, sigma_t=self.qubit_ramp_ns)
            sync()
        # --------------------------------------------------------------------

        # setup sBs step -----------------------------------------------------
        CD_compiler_kwargs = dict(cal_dir=self.cal_dir)
        CD_params_func_kwargs = dict(name='CD_params_fixed_tau_from_cal', tau_ns=self.tau_ns)
        C = SBS_simple_compiler(CD_compiler_kwargs, CD_params_func_kwargs)

        cavity_pulse, qubit_pulse = C.make_pulse(self.eps1/2.0, self.eps2/2.0, -1j*self.beta)

        cavity_pulse = {'x': cavity_pulse, 'p': (-cavity_pulse[1], cavity_pulse[0])}

        def sbs_step(s):
            sync()
            self.cavity.array_pulse(*cavity_pulse[s])
            self.qubit.array_pulse(*qubit_pulse)
            sync()
        # --------------------------------------------------------------------

        def step(s):
            sbs_step(s)
            cooling()


        with system.wigner_tomography(*self.disp_range, result_name='m2'):
            readout(m0='se')
            sync()
            with Repeat(self.reps):
                step('x')
                step('p')
            sync()
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
