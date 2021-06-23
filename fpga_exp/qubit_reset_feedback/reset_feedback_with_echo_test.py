# -*- coding: utf-8 -*-
"""
Created on Sat May 29 14:39:30 2021

@author: qulab
"""
from init_script import *

class reset_feedback_with_echo_test(FPGAExperiment):
    loop_delay = IntParameter(500e3)
    alpha = FloatParameter(0)

    feedback_delay = IntParameter(0)
    final_delay = IntParameter(0)
    echo_delay = IntParameter(0)

    def sequence(self):

        @subroutine
        def reset_with_echo():
            sync()
            delay(self.echo_delay, channel=qubit.chan, round=True)
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
            delay(self.final_delay, round=True)
            sync()
        
        readout(m0='se')
        qubit.pi2_pulse()
        sync()
        cavity.displace(self.alpha)
        sync()
        reset_with_echo()
        sync()
        readout(m1='se')
        delay(self.loop_delay)


    def process_data(self):
        
        not_ge = self.results['m0'].multithresh()[2]

        postselected = self.results['m1'].postselect(not_ge, [0])[0]
        self.results['m1_postselected_m0'] = postselected
        self.results['m1_postselected_m0'].ax_data = self.results['m1'].ax_data
        self.results['m1_postselected_m0'].labels = self.results['m1'].labels

        postselected = self.results['m0'].postselect(not_ge, [0])[0]
        self.results['m0_postselected_m0'] = postselected
        self.results['m0_postselected_m0'].ax_data = self.results['m1'].ax_data
        self.results['m0_postselected_m0'].labels = self.results['m1'].labels