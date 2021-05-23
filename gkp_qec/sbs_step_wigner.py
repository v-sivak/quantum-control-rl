# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:55:46 2021

@author: qulab
"""
from init_script import *
from gkp_exp.CD_gate.conditional_displacement_compiler import SBS_simple_compiler
import numpy as np

class sbs_step_wigner(FPGAExperiment):
    disp_range = RangeParameter((-4.0, 4.0, 101))
    s_tau_ns = IntParameter(20)
    b_tau_ns = IntParameter(50)

    eps1 = FloatParameter(0.2)
    eps2 = FloatParameter(0.2)
    beta = FloatParameter(2.5066) # np.sqrt(2*np.pi)

    s_CD_cal_dir = StringParameter('')
    b_CD_cal_dir = StringParameter('')


    def sequence(self):
        CD_compiler_kwargs = dict(qubit_pulse_pad=4)
        s_CD_params_func_kwargs = dict(name='CD_params_fixed_tau_from_cal',
                                       tau_ns=self.s_tau_ns, cal_dir=self.s_CD_cal_dir)
        b_CD_params_func_kwargs = dict(name='CD_params_fixed_tau_from_cal',
                                       tau_ns=self.b_tau_ns, cal_dir=self.b_CD_cal_dir)
        C = SBS_simple_compiler(CD_compiler_kwargs,
                                s_CD_params_func_kwargs, b_CD_params_func_kwargs)

        cavity_pulse, qubit_pulse = C.make_pulse(
                self.eps1/2.0, self.eps2/2.0, -1j*self.beta)


        self.cavity = cavity_1

        with system.wigner_tomography(*self.disp_range):
            readout(init='se')
            sync()
            self.cavity.array_pulse(*cavity_pulse)
            qubit.array_pulse(*qubit_pulse)
            sync()
            delay(24)
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
