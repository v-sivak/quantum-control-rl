# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:21:36 2021

@author: Vladimir Sivak
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP

class sbs_stabilizer_reward_fpga(FPGAExperiment, GKP):
    export_txt = False
    save_log = False
    
    """ GKP stabilization with SBS protocol. """
    loop_delay = FloatParameter(4e6)
    batch_size = IntParameter(10)
    opt_file = StringParameter('')
    xp_rounds = IntParameter(15)
    tau_stabilizer = IntParameter(50)
    echo_delay = IntParameter(880)
    final_delay = IntParameter(92)

    def sequence(self):
        # load SBS pulse sequences from file
        params = np.load(self.opt_file, allow_pickle=True)
        self.stabilizers = params['stabilizers']
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]

        self.readout, self.qubit, self.cavity = readout, qubit, cavity

        reset = lambda: self.reset_feedback_with_echo(self.echo_delay, self.final_delay)

        def sbs_step(i, s):
            """
            Args:
                i (int): batch index
                s (str): stabilizer direction, either 'x' or 'p'
            """
            phase = dict(x=0.0, p=np.pi/2.0)
            sync()
            self.cavity.array_pulse(*self.cavity_pulses[i], phase=phase[s])
            self.qubit.array_pulse(*self.qubit_pulses[i])
            sync()

        def control_circuit(i):
            sync()
            with Repeat(self.xp_rounds):
                sbs_step(i, 'x')
                reset()
                sbs_step(i, 'p')
                reset()
            sync()

        @subroutine
        def reward_circuit(s):
            self.reset_feedback_with_echo(self.echo_delay, self.final_delay, log=True, res_name='m1_'+str(s))
            self.displacement_phase_estimation(s, self.tau_stabilizer, res_name='m2_'+str(s))
            delay(self.loop_delay)

        # experience collection loop
        for i in range(self.batch_size):
            for s in self.stabilizers:
                control_circuit(i)
                reward_circuit(s)

