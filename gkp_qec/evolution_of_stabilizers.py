# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:37:07 2021

@author: qulab
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP

class evolution_of_stabilizers(FPGAExperiment, GKP):

    reps = IntParameter(5)

    # Baptiste sbs stabilization parameters
    s_tau_ns = IntParameter(20)
    b_tau_ns = IntParameter(50)
    eps1 = FloatParameter(0.2)
    eps2 = FloatParameter(0.2)
    beta = FloatParameter(2.5066) # np.sqrt(2*np.pi)
    cal_dir = StringParameter('')

    # Feedback cooling parameters
    echo_delay = IntParameter(0)
    final_delay = IntParameter(0)

    loop_delay = IntParameter(4e6)


    def sequence(self):

        self.readout = readout
        self.qubit = qubit
        self.cavity = cavity

        reset = lambda: self.reset_feedback_with_echo(self.echo_delay, self.final_delay)

        sbs_step = self.sbs(self.eps1, self.eps2, self.beta,
                            self.s_tau_ns, self.b_tau_ns, self.cal_dir)

        def step(s):
            sbs_step(s)
            reset()

        def exp(reps):
            sync()
            with Repeat(reps):
                step('x')
                step('p')
            sync()

        stabilizer_phase_estimation = self.stabilizer_phase_estimation(
                self.b_tau_ns, self.cal_dir)

        for s in ['x', 'p']:
            for reps in range(self.reps):
                exp(reps)
                stabilizer_phase_estimation(s)
                delay(self.loop_delay)


    def process_data(self):
        for s in ['x','p']:
            self.results[s+'_sigma_z'] = 1-2*self.results[s].threshold()
            self.results[s+'_sigma_z'].ax_data = self.results[s].ax_data
            self.results[s+'_sigma_z'].labels = self.results[s].labels


    def plot(self, fig, data):
        
        ax = fig.add_subplot(111)
        for s in ['x','p']:
            ax.plot(self.results[s+'_sigma_z'].ax_data[1], 
                    self.results[s+'_sigma_z'].data.mean(axis=0),
                    marker='.')