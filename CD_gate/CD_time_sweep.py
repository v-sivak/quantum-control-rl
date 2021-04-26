# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:38:45 2021

@author: qulab
"""
import numpy as np
from init_script import *
from conditional_displacement_compiler import ConditionalDisplacementCompiler

class CD_time_sweep(FPGAExperiment):
    """
    This experiment for each gate time, finds the parameters of the CD gate
    (amplitude and phases), does the gate and then displacement to return
    cavity to vacuum.

    """
    beta = FloatParameter(1.)
    cal_dir = StringParameter(r'C:\code\gkp_exp\CD_gate\storage_rotation_freq_vs_nbar')
    tau_range_ns = RangeParameter((400,50,21))
    loop_delay = FloatParameter(4e6)
    flip_qubit = BoolParameter(False)

    def sequence(self):
        return_phase = np.pi/2.0 if not self.flip_qubit else -np.pi/2.0
        tau_ns = np.linspace(*self.tau_range_ns)

        C = ConditionalDisplacementCompiler(cal_dir=self.cal_dir)

        for tau in tau_ns:
            tau, alpha, phi_g, phi_e = C.CD_params_fixed_tau(self.beta, tau)
            self.cavity_pulse, self.qubit_pulse = C.make_pulse(tau, alpha, phi_g, phi_e)
            extra_phase = phi_g + phi_e

            if self.flip_qubit:
                qubit.flip()
            sync()
            qubit.array_pulse(*self.qubit_pulse)
            cavity.array_pulse(*self.cavity_pulse)
            sync()
            cavity.displace(self.beta/2.0, phase=return_phase-extra_phase)
            sync()
            qubit.flip(selective=True)
            sync()
            readout()
            delay(self.loop_delay)

    def process_data(self):
        if self.blocks_processed >= 1:
            self.results['default'].ax_data[1] = np.linspace(*self.tau_range_ns)
            self.results['default'].labels[1] = 'Delay (ns)'
