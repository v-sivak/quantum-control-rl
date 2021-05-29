# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:50:52 2021

@author: qulab
"""


from init_script import *

class reset_feedback_test(FPGAExperiment):
    loop_delay = IntParameter(500e3)

    def sequence(self):
        qubit.pi2_pulse()

        sync()
        readout(wait_result=True, log=False)
        sync()
        if_then_else(qubit.measured_low(), 'wait', 'flip')
        label_next('flip')
        qubit.flip()
        goto('continue')
        label_next('wait')
        delay(qubit.pulse.length)
        label_next('continue')
        sync()

        readout()
        delay(self.loop_delay)
