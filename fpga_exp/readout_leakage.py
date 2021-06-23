# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:56:47 2021

@author: qulab
"""
from init_script import *
import numpy as np
import matplotlib.pyplot as plt


class readout_leakage(FPGAExperiment):
    """ Do two consequtive readouts and post-select the 2nd based on the 1st.
    """
    loop_delay = IntParameter(1e6)
    read_delay = IntParameter(1e3)
    alpha = FloatParameter(1.0)

    def sequence(self):
        qubit.pi2_pulse()
        sync()
        cavity.displace(self.alpha)
        sync()
        readout(m0='se')
        delay(self.read_delay)
        readout(m1='se')
        delay(self.loop_delay)

    def process_data(self):
        for i in range(3):
            init_state = self.results['m0'].multithresh()[i]
            postselected = self.results['m1'].postselect(init_state, [1])[0]
            self.results['m1_postselected_'+str(i)] = postselected
            self.results['m1_postselected_'+str(i)].ax_data = self.results['m1'].ax_data
            self.results['m1_postselected_'+str(i)].labels = self.results['m1'].labels
