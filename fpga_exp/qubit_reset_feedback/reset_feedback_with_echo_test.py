# -*- coding: utf-8 -*-
"""
Created on Sat May 29 14:39:30 2021

@author: qulab
"""
from init_script import *

class reset_feedback_with_echo_test(FPGAExperiment):
    loop_delay = IntParameter(500e3)

    feedback_delay = IntParameter(0)
    final_delay = IntParameter(0)
    echo_delay = IntParameter(0)

    def sequence(self):

        @subroutine
        def reset_with_echo():
            sync()
            delay(self.echo_delay, channel=qubit.chan)
            qubit.flip() # echo pulse
            readout(wait_result=True, log=False, sync_at_beginning=False)
            sync()
            delay(self.feedback_delay, round=True)
            if_then_else(qubit.measured_low(), 'flip', 'wait')
            label_next('flip')
            qubit.flip()
            goto('continue')
            label_next('wait')
            delay(qubit.pulse.length)
            label_next('continue')
            delay(self.final_delay)
            sync()

        qubit.pi2_pulse()
        reset_with_echo()
        readout()
        delay(self.loop_delay)
