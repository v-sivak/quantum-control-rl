# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:21:36 2021

@author: Vladimir Sivak
"""
from init_script import *
import numpy as np

class gkp_prep_ECD_stabilizers_fpga(FPGAExperiment):
    """ Learn preparation of the GKP state with ECD control using stabilizer
        values as rewards. 
    """
    export_txt = False
    save_log = False

    batch_size = IntParameter(10)
    opt_file = StringParameter('')


    def sequence(self):
        # load SBS pulse sequences from file
        params = np.load(self.opt_file, allow_pickle=True)
        self.stabilizers = params['stabilizers']
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]
        
        # setup mode objects
        gkp.readout, gkp.qubit, gkp.cavity = readout, qubit, cavity

        def control_circuit(i):
            sync()
            qubit.array_pulse(*self.qubit_pulses[i])
            cavity.array_pulse(*self.cavity_pulses[i])
            sync()

        @subroutine
        def reward_circuit(s):
            gkp.reset_feedback_with_echo(gkp.echo_delay, gkp.final_delay, log=True, res_name='m1_'+str(s))
            gkp.displacement_phase_estimation(s, gkp.t_stabilizer_ns, res_name='m2_'+str(s))
            delay(gkp.loop_delay)

        # experience collection loop
        for i in range(self.batch_size):
            for s in self.stabilizers:
                control_circuit(i)
                reward_circuit(s)

