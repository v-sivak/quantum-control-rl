# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:40:24 2021

@author: qulab
"""
from init_script import *
from conditional_displacement_compiler import ConditionalDisplacementCompiler

class CD_cat_wigner(FPGAExperiment):
    disp_range = RangeParameter((-4.0, 4.0, 101))
    tau = IntParameter(100)
    alpha = FloatParameter(6.0)

    def sequence(self):
        C = ConditionalDisplacementCompiler()
        cavity_pulse, qubit_pulse = C.make_pulse(self.tau, self.alpha, 0., 0.)

        with system.wigner_tomography(*self.disp_range):
            qubit.pi2_pulse(phase=np.pi/2)
            sync()
            cavity.array_pulse(*cavity_pulse)
            qubit.array_pulse(*qubit_pulse)
            sync()
            qubit.pi2_pulse(phase=np.pi/2)
            sync()
            readout(msmt='se')
            qubit.flip()
            delay(24+252+24+1000+12)


    def process_data(self):
        msmt = self.results['msmt'].threshold()
        for i in [0,1]:
            self.results['postselected'+str(i)] = self.results['default'].postselect(msmt, [i])[0].thresh_mean().data
            self.results['postselected'+str(i)].ax_data = self.results['msmt'].ax_data[1:]
            self.results['postselected'+str(i)].labels = self.results['msmt'].labels[1:]
