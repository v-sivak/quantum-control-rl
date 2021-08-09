from init_script import *
import numpy as np
from CD_gate.conditional_displacement_compiler import ECD_control_simple_compiler

class fock1_T1(FPGAExperiment):
    points = IntParameter(41)
    t_max = FloatParameter(1e5)
    loop_delay=IntParameter(4e6)
    delay_repeats = IntParameter(2)
    fit_func = 'exp_decay'
    fit_fmt = {'tau': ('%.1f us', 1e3)}
    
    fock1_file = StringParameter('')

    def sequence(self):

        data = np.load(self.fock1_file, allow_pickle=True)
        c_pulse = (data['c_pulse'].real, data['c_pulse'].imag)
        q_pulse = (data['q_pulse'].real, data['q_pulse'].imag)
        
        
        with scan_length(0, self.t_max, self.points, 
                         axis_scale=float(self.delay_repeats)) as dynlen:
            delay(100)
            
            sync()
            cavity.array_pulse(*c_pulse)
            qubit.array_pulse(*q_pulse)
            sync()
            
            for i in range(self.delay_repeats):
                delay(dynlen)
            
            qubit.flip(selective=True, detune=-cavity.chi)
            readout()
            delay(self.loop_delay)

    def update(self):
        cavity.t1 = self.fit_params['tau']