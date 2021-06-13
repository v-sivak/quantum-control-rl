# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:21:36 2021

@author: Vladimir Sivak
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP

class sbs_stabilizer_reward_v3_fpga(FPGAExperiment, GKP):
    export_txt = False
    save_log = False
    
    """ GKP stabilization with SBS protocol. """
    loop_delay = FloatParameter(4e6)
    batch_size = IntParameter(10)
    opt_file = StringParameter('')
    xp_rounds = IntParameter(15)
    tau_stabilizer = IntParameter(50)
    cal_dir = StringParameter('')
    echo_delay = IntParameter(880)
    final_delay = IntParameter(92)
    stabilizers = StringParameter('x,p')

    def sequence(self):
        # load SBS pulse sequences from file
        params = np.load(self.opt_file, allow_pickle=True)
        qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]
        
        echo_delays = params['echo_delay']
        feedback_delays = params['feedback_delay']
        final_delays = params['final_delay']

        self.readout, self.qubit, self.cavity = readout, qubit, cavity

        def sbs_step(i, s):
            """
            Args:
                i (int): batch index
                s (str): stabilizer direction, either 'x' or 'p'
            """
            phase = dict(x=0.0, p=np.pi/2.0)
            sync()
            self.cavity.array_pulse(*cavity_pulses[i], phase=phase[s])
            self.qubit.array_pulse(*qubit_pulses[i])
            sync()
        
        def reset(i):
            self.reset_feedback_with_echo(
                    echo_delays[i], final_delays[i], feedback_delays[i])
        
        def control_circuit(i):
            sync()
            with Repeat(self.xp_rounds):
                sbs_step(i, 'x')
                reset(i)
                sbs_step(i, 'p')
                reset(i)
            sync()

        stabilizer_phase_estimation = self.stabilizer_phase_estimation(
                self.tau_stabilizer, self.cal_dir)

        @subroutine
        def reward_circuit(s):
            stabilizer_phase_estimation(s)
            delay(self.loop_delay)

        # experience collection loop
        for i in range(self.batch_size):
            for s in self.stabilizers.split(','):
                control_circuit(i)
                reward_circuit(s)

