# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:40:24 2021

@author: qulab
"""
from init_script import *
from conditional_displacement_compiler import ConditionalDisplacementCompiler

class CD_cat_wigner(FPGAExperiment):
    disp_range = RangeParameter((-4.0, 4.0, 101))

    # Parameters of the CD
    tau = IntParameter(50)
    beta = FloatParameter(1.0)
    cal_dir = StringParameter('')

    def sequence(self):
        C = ConditionalDisplacementCompiler(qubit_pulse_pad=4)
        (tau, alpha, phi_g, phi_e) = C.CD_params_fixed_tau_from_cal(self.beta, self.tau, self.cal_dir)
        cavity_pulse, qubit_pulse = C.make_pulse(tau, alpha, 0., 0.)

        with system.wigner_tomography(*self.disp_range):
            readout(init='se')
            sync()
            qubit.pi2_pulse(phase=np.pi/2)
            sync()
            cavity.array_pulse(*cavity_pulse)
            qubit.array_pulse(*qubit_pulse)
            sync()
            qubit.pi2_pulse(phase=np.pi/2)
            sync()
            readout(msmt='se')


    def process_data(self):
        init = self.results['init'].threshold()
        msmt = self.results['msmt'].threshold()
        for i in [0,1]:
            for j in [0,1]:
                result = self.results['default'].postselect(init, [i])[0].postselect(msmt, [j])[0]
                self.results['postselected'+str(i)+str(j)] = 1 - 2*result.thresh_mean().data
                self.results['postselected'+str(i)+str(j)].ax_data = self.results['msmt'].ax_data[1:]
                self.results['postselected'+str(i)+str(j)].labels = self.results['msmt'].labels[1:]
                self.results['postselected'+str(i)+str(j)].vmin = -1
                self.results['postselected'+str(i)+str(j)].vmax = +1
