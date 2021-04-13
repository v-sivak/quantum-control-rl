# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:03:29 2021

@author: qulab
"""
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler
from init_script import *

class state_prep_wigner_tomography(FPGAExperiment):
    """
    Load ECD control sequence from file and take the Wigner of the state.
    
    """
    loop_delay = FloatParameter(4e6)
    disp_range = RangeParameter((-4.0, 4.0, 41))
    tau_ns = IntParameter(24)
    filename = StringParameter(r'C:\code\gkp_exp\state_prep\fock4.npz')

    def sequence(self):
        data = np.load(self.filename)
        beta, phi = data['beta'], data['phi']
        C = ECD_control_simple_compiler(tau_ns=self.tau_ns)
        self.c_pulse, self.q_pulse = C.make_pulse(beta, phi)

        def ECDC_sequence():
            sync()
            qubit.array_pulse(self.q_pulse.real, self.q_pulse.imag)
            cavity.array_pulse(self.c_pulse.real, self.c_pulse.imag)
            sync()

        with system.wigner_tomography(*self.disp_range, result_name='m2'):
            ECDC_sequence()
            readout(m1='se')
            
    
    def process_data(self):
        # post-select on m1='g' outcomes
        init_state = self.results['m1'].threshold()
        postselected = self.results['m2'].postselect(init_state, [0])[0]
        self.results['m2_postselected'] = 1 - 2*postselected.threshold()
        self.results['m2_postselected'].ax_data = self.results['m1'].ax_data
        self.results['m2_postselected'].labels = self.results['m1'].labels
