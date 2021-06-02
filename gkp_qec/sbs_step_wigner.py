# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:55:46 2021

@author: qulab
"""
from init_script import *
from gkp_exp.CD_gate.conditional_displacement_compiler import SBS_simple_compiler, SBS_Alec_compiler
import numpy as np

class sbs_step_wigner(FPGAExperiment):
    disp_range = RangeParameter((-4.0, 4.0, 101))
    s_tau_ns = IntParameter(20)
    b_tau_ns = IntParameter(50)

    eps1 = FloatParameter(0.2)
    eps2 = FloatParameter(0.2)
    beta = FloatParameter(2.5066) # np.sqrt(2*np.pi)

    cal_dir = StringParameter('')


    def sequence(self):
        CD_compiler_kwargs = dict(qubit_pulse_pad=0)
        C = SBS_simple_compiler(CD_compiler_kwargs, self.cal_dir)

        #ECD_control_kwargs = dict(alpha_CD=8.0, buffer_time=4)
        #C = SBS_Alec_compiler(ECD_control_kwargs)

        cavity_pulse, qubit_pulse = C.make_pulse(
                self.eps1/2.0, self.eps2/2.0, -1j*self.beta,
                self.s_tau_ns, self.b_tau_ns)


        self.cavity = cavity

        with system.wigner_tomography(*self.disp_range):
            readout(m0='se')
            sync()
            self.cavity.array_pulse(*cavity_pulse)
            qubit.array_pulse(*qubit_pulse)
            sync()
            delay(24)
            readout(m1='se')
            sync()


    def process_data(self):
        m0 = self.results['m0'].threshold()
        m1 = self.results['m1'].threshold()
        for i in [0,1]:
            for j in [0,1]:
                result = self.results['default'].postselect(m0, [i])[0].postselect(m1, [j])[0]
                self.results['postselected'+str(i)+str(j)] = result.thresh_mean().data
                self.results['postselected'+str(i)+str(j)].ax_data = self.results['m1'].ax_data[1:]
                self.results['postselected'+str(i)+str(j)].labels = self.results['m1'].labels[1:]
