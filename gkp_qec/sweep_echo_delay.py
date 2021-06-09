# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:31:34 2021

@author: qulab
"""


# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:37:07 2021

@author: qulab
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP

class sweep_echo_delay(FPGAExperiment, GKP):

    xp_rounds = IntParameter(15)

    # sbs stabilization parameters
    s_tau_ns = IntParameter(20)
    b_tau_ns = IntParameter(50)
    ECD_filename = StringParameter(r'Y:\tmp\for Vlad\from_☺vlad\000500sbs_run3.npz')
    cal_dir = StringParameter(r'D:\DATA\exp\2021-05-13_cooldown\CD_fixed_time_amp_cal')

    # Feedback cooling parameters
    final_delay = IntParameter(0)
    echo_delay_range = RangeParameter((0,200,8))

    loop_delay = IntParameter(4e6)

    def sequence(self):

        self.readout = readout
        self.qubit = qubit
        self.cavity = cavity

        stabilizer_phase_estimation = self.stabilizer_phase_estimation(self.b_tau_ns, self.cal_dir)
        sbs_step = self.load_sbs_sequence(self.s_tau_ns, self.b_tau_ns, self.ECD_filename, self.cal_dir)


        for echo_delay in np.linspace(*self.echo_delay_range):

            reset = lambda: self.reset_feedback_with_echo(echo_delay, self.final_delay)

            def control_circuit():
                sync()
                with Repeat(self.xp_rounds):
                    sbs_step('x')
                    reset()
                    sbs_step('p')
                    reset()
                sync()

            control_circuit()
            stabilizer_phase_estimation('p')
            delay(self.loop_delay)



    def process_data(self):

        self.results['sigma_z'] = 1-2*self.results['p'].threshold()
        self.results['sigma_z'].ax_data = self.results['p'].ax_data
        self.results['sigma_z'].labels = self.results['p'].labels
