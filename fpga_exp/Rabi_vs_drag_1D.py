# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 19:18:02 2021

@author: qulab
"""
from init_script import *
import numpy as np
import matplotlib.pyplot as plt


class Rabi_vs_drag_1D(FPGAExperiment):

    drag_range = RangeParameter((-10,10,21))
    loop_delay = FloatParameter(1e6)

    def sequence(self):

        pulse = qubit.pulse

        for drag in np.linspace(*self.drag_range):
            pulse.detune = 0
            pulse.drag = drag

            readout(m0='se')
            pulse()
            readout(m1='se')
            delay(self.loop_delay)


    def process_data(self):
        init_state_g = self.results['m0'].multithresh()[0]
        postselected = self.results['m1'].postselect(init_state_g, [1])[0]
        self.results['m1_postselected_m0'] = postselected        
        self.results['m1_postselected_m0'].ax_data = self.results['m1'].ax_data
        self.results['m1_postselected_m0'].labels = self.results['m1'].labels        