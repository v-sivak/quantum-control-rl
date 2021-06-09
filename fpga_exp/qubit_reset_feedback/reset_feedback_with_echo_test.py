# -*- coding: utf-8 -*-
"""
Created on Sat May 29 14:39:30 2021

@author: qulab
"""
from init_script import *

class reset_feedback_with_echo_test(FPGAExperiment):
    delay_length = IntParameter(0)
    loop_delay = IntParameter(500e3)

    def sequence(self):

        @subroutine
        def reset_with_echo():
            sync()
            readout(wait_result=True, log=False, **{'m0':'se'})
            sync()
            qubit.flip() # echo pulse
            sync()
            delay(self.delay_length)
            if_then_else(qubit.measured_low(), 'flip', 'wait')
            label_next('flip')
            qubit.flip()
            goto('continue')
            label_next('wait')
            delay(qubit.pulse.length)
            label_next('continue')
            delay(24)
            sync()

        qubit.pi2_pulse()
        reset_with_echo()
        readout()
        delay(self.loop_delay)
