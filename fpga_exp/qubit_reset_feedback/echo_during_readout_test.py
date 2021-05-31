# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:56:30 2021

@author: qulab
"""
from init_script import *

class echo_during_readout_test(FPGAExperiment):
    echo_delay = IntParameter(0)
    loop_delay = IntParameter(500e3)

    def sequence(self):
        
        @subroutine
        def reset_with_echo():
            sync()
            delay(self.echo_delay, channel=qubit.chan)
            qubit.flip() # echo pulse
            readout(wait_result=True, log=False, sync_at_beginning=False)
            sync()
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