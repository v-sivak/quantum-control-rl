# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:55:46 2021

@author: qulab
"""
from init_script import *
from gkp_exp.CD_gate.conditional_displacement_compiler import sBs_compiler
import numpy as np

class sbs_step_wigner(FPGAExperiment):

    cal_dir = StringParameter(r'C:\code\gkp_exp\CD_gate\storage_rotation_freq_vs_nbar')
    disp_range = RangeParameter((-4.0, 4.0, 101))
    tau_small = IntParameter(12)
    tau_big = IntParameter(50)
    eps1 = FloatParameter(0.2)
    eps2 = FloatParameter(0.2)



    def sequence(self):

        C = sBs_compiler(cal_dir=self.cal_dir, tau_small=self.tau_small, tau_big=self.tau_big)
        cavity_pulse, qubit_pulse = C.make_pulse(self.eps1, self.eps2, np.sqrt(2*np.pi))

        with system.wigner_tomography(*self.disp_range):
            readout(init='se')
            sync()
            cavity.array_pulse(*cavity_pulse)
            qubit.array_pulse(*qubit_pulse)
            sync()
            readout(msmt='se')
            sync()


    def process_data(self):
        init = self.results['init'].threshold()
        msmt = self.results['msmt'].threshold()
        for i in [0,1]:
            for j in [0,1]:
                result = self.results['default'].postselect(init, [i])[0].postselect(msmt, [j])[0]
                self.results['postselected'+str(i)+str(j)] = result.thresh_mean().data
                self.results['postselected'+str(i)+str(j)].ax_data = self.results['msmt'].ax_data[1:]
                self.results['postselected'+str(i)+str(j)].labels = self.results['msmt'].labels[1:]
