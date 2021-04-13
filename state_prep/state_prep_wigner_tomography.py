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
    alpha_abs = FloatParameter(8.0)
    filename = StringParameter(r'C:\code\gkp_exp\state_prep\fock4.npz')

    def sequence(self):
        data = np.load(self.filename)
        beta, phi = data['beta'], data['phi']
        C = ECD_control_simple_compiler(tau_ns=self.tau_ns)
#        C = ECD_control_simple_compiler(alpha_abs=self.alpha_abs)
        self.c_pulse, self.q_pulse = C.make_pulse(beta, phi)

        def ECDC_sequence():
            sync()
            qubit.array_pulse(self.q_pulse.real, self.q_pulse.imag)
            cavity.array_pulse(self.c_pulse.real, self.c_pulse.imag)
            sync()

        with system.wigner_tomography(*self.disp_range, result_name='m2'):
            readout(m0='se')
            ECDC_sequence()
            readout(m1='se')
            
    
    def process_data(self):
        # post-select on m0='g' outcomes
        init_state = self.results['m0'].threshold()
        postselected = self.results['m2'].postselect(init_state, [0])[0]
        self.results['m2_postselected_m0'] = 1 - 2*postselected.threshold()
        self.results['m2_postselected_m0'].ax_data = self.results['m2'].ax_data
        self.results['m2_postselected_m0'].labels = self.results['m2'].labels
        self.results['m2_postselected_m0'].vmin = -1
        self.results['m2_postselected_m0'].vmax = 1    


        # post-select on m1='g' outcomes
        verification = self.results['m1'].threshold()
        postselected = self.results['m2_postselected_m0'].postselect(verification, [0])[0]
        self.results['m2_postselected_m0_m1'] = postselected.threshold()
        self.results['m2_postselected_m0_m1'].ax_data = self.results['m2'].ax_data
        self.results['m2_postselected_m0_m1'].labels = self.results['m2'].labels
        self.results['m2_postselected_m0_m1'].vmin = -1
        self.results['m2_postselected_m0_m1'].vmax = 1