# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:56:47 2021

@author: qulab
"""
from init_script import *
import numpy as np
import matplotlib.pyplot as plt


class readout_repeat(FPGAExperiment):
    """ Simply do multiple readouts in a row to see how the leakage outside 
        g/e manifold accumulates.
    """
    loop_delay = FloatParameter(1e6)
    reps = IntParameter(15)
    read_delay = IntParameter(1000)
    alpha = FloatParameter(0.0)

    def sequence(self):
        cavity.displace(self.alpha)
        with Repeat(self.reps):
            sync()
            readout()
            delay(self.read_delay)
        delay(self.loop_delay)