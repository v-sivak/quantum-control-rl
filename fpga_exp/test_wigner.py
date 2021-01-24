# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:08:01 2021

@author: qulab
"""
from init_script import *

class test_wigner(FPGAExperiment):
    alpha = FloatParameter(2.0)
    disp_range = RangeParameter((-4.0, 4.0, 101))
    def sequence(self):
        with system.wigner_tomography(*self.disp_range):
            cavity.displace(self.alpha)
