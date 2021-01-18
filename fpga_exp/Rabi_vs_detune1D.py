# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:24:37 2021

@author: qulab
"""
from init_script import *
import numpy as np
import matplotlib.pyplot as plt


class Rabi_vs_detune1D(FPGAExperiment):

    selective = BoolParameter(False)
    detune = RangeParameter((-100e3, 100e3, 11))
    loop_delay = FloatParameter(1e6)
    ssb_base = FloatParameter(-50e6)

    def sequence(self):
        pulse = qubit.selective_pulse if self.selective else qubit.pulse
        with qubit.scan_detune(*self.detune, ssb0=self.ssb_base):
            sync()
            system.cool_qubit()
            sync()
            pulse()
            sync()
            readout()
            delay(self.loop_delay)