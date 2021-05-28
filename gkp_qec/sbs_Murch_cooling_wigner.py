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
    additional_delay = IntParameter(0)


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


        reset = lambda: self.cooling_Murch(self.cool_duration_ns, self.qubit_ramp_ns, 
                            self.readout_ramp_ns, self.qubit_amp, self.readout_amp)
        # --------------------------------------------------------------------
        # --------------------------------------------------------------------


        # setup SBS step -----------------------------------------------------
        CD_compiler_kwargs = dict(qubit_pulse_pad=4)
        s_CD_params_func_kwargs = dict(name='CD_params_fixed_tau_from_cal',
                                       tau_ns=self.s_tau_ns, cal_dir=self.s_CD_cal_dir)
        b_CD_params_func_kwargs = dict(name='CD_params_fixed_tau_from_cal',
                                       tau_ns=self.b_tau_ns, cal_dir=self.b_CD_cal_dir)
        C = SBS_simple_compiler(CD_compiler_kwargs,
                                s_CD_params_func_kwargs, b_CD_params_func_kwargs)

        cavity_pulse, qubit_pulse = C.make_pulse(1j*self.eps1/2.0, 1j*self.eps2/2.0, self.beta)

        qubit_pulse_sbs = qubit_pulse
        cavity_pulse_sbs = {'x': cavity_pulse,
                            'p': (-cavity_pulse[1], cavity_pulse[0])}

        def sbs_step(s):
            sync()
            self.cavity.array_pulse(*cavity_pulse_sbs[s])
            self.qubit.array_pulse(*qubit_pulse_sbs)
            sync()
        # --------------------------------------------------------------------
        # --------------------------------------------------------------------

        def snap():
            delay(self.additional_delay)

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


    @subroutine
    def cooling_Murch(self, cool_duration_ns, qubit_ramp_ns, readout_ramp_ns,
                qubit_amp, readout_amp):
        """
        Autonomous qubit cooling based on this Murch paper:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.183602

        Args:
            cool_duration_ns (int): how long in [ns] to hold the constant
                Rabi drive on the qubit after ramping it up.
            qubit_ramp_ns (int): duration in [ns] of the qubit Rabi drive
                ramp up/down.
            readout_ramp_ns (int): duration in [ns] of the detuned readout
                drive ramp up/down. This can typically be shorter than the
                qubit ramp because the pulse is far detuned.
            qubit_amp (float): amplitude of the qubit Rabi pulse.
            readout_amp (float): amplitude of the detuned readout pulse.
        """
        # calculate the duration of the constant part of the readout pulse
        readout_pump_time = cool_duration_ns+2*qubit_ramp_ns-2*readout_ramp_ns

        sync()
        self.readout_detuned.smoothed_constant_pulse(
                readout_pump_time, amp=readout_amp, sigma_t=readout_ramp_ns)
        self.qubit_detuned.smoothed_constant_pulse(
                cool_duration_ns, amp=qubit_amp, sigma_t=qubit_ramp_ns)
        sync()


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
