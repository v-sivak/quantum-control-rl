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
        data = np.load(self.filename, allow_pickle=True)
        beta, phi = data['beta'], data['phi']
        C = ECD_control_simple_compiler(tau_ns=self.tau_ns)
#        C = ECD_control_simple_compiler(alpha_abs=self.alpha_abs)
        self.c_pulse, self.q_pulse = C.make_pulse(beta, phi)

#        data = np.load(r'Y:\tmp\for Vlad\from_vlad\vlad_params_fock_4_alpha_7.000.npz')
#        self.q_pulse = data['qubit_dac_pulse']
#        self.c_pulse = data['cavity_dac_pulse']

        def ECDC_sequence():
            sync()
            qubit.array_pulse(self.q_pulse.real, self.q_pulse.imag)
            cavity.array_pulse(self.c_pulse.real, self.c_pulse.imag)
            sync()

        # map odd parity to 'e'
        with system.wigner_tomography(*self.disp_range, result_name='g_m2',
                                      invert_axis=False):
            readout(g_m0='se')
            ECDC_sequence()
            readout(g_m1='se')

        # map odd parity to 'g'
        with system.wigner_tomography(*self.disp_range, result_name='e_m2',
                                      invert_axis=True):
            readout(e_m0='se')
            ECDC_sequence()
            readout(e_m1='se')


    def process_data(self):
        sign = dict(g=1, e=-1)

        for s in ['g', 'e']:
            # post-select on m0='g'
            init_state = self.results[s+'_m0'].threshold()
            postselected = self.results[s+'_m2'].postselect(init_state, [0])[0]
            self.results[s+'_m2_postselected_m0'] = sign[s]*(1 - 2*postselected.threshold())
            self.results[s+'_m2_postselected_m0'].ax_data = self.results[s+'_m2'].ax_data
            self.results[s+'_m2_postselected_m0'].labels = self.results[s+'_m2'].labels
            self.results[s+'_m2_postselected_m0'].vmin = -1
            self.results[s+'_m2_postselected_m0'].vmax = 1

            # post-select on m1='g' outcomes
            verification = self.results[s+'_m1'].threshold()
            postselected = self.results[s+'_m2_postselected_m0'].postselect(verification, [0])[0]
            self.results[s+'_m2_postselected_m0_m1'] = postselected.threshold()
            self.results[s+'_m2_postselected_m0_m1'].ax_data = self.results[s+'_m2'].ax_data
            self.results[s+'_m2_postselected_m0_m1'].labels = self.results[s+'_m2'].labels
            self.results[s+'_m2_postselected_m0_m1'].vmin = -1
            self.results[s+'_m2_postselected_m0_m1'].vmax = 1
