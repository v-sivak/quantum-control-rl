# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:42:17 2021

@author: Vladimir Sivak
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP

class sbs_stabilizer_reward_mixer_updates_fpga(FPGAExperiment, GKP):
    export_txt = False
    save_log = False

    """ GKP stabilization with SBS protocol. """
    loop_delay = FloatParameter(4e6)
    batch_size = IntParameter(10)
    opt_file = StringParameter('')
    xp_rounds = IntParameter(15)
    tau_stabilizer = IntParameter(50)
    echo_delay = IntParameter(880)
    final_delay = IntParameter(0)
    t_mixer_calc = IntParameter(600)

    def sequence(self):
        """
        This sequence has 3 components:
            1) SBS step (ECDC sequence)
            2) qubit reset through feedback with echo pulse
            3) cavity phase update with dynamic mixer

        """
        self.readout, self.qubit, self.cavity = readout, qubit, cavity

        # load experimental parameters from file
        params = np.load(self.opt_file, allow_pickle=True)
        self.stabilizers = params['stabilizers']
        self.cavity_phases = params['cavity_phases']
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]


        def sbs_step(i):
            sync()
            self.cavity.array_pulse(*self.cavity_pulses[i], amp='dynamic')
            self.qubit.array_pulse(*self.qubit_pulses[i])
            sync()

        reset = lambda: self.reset_feedback_with_echo(self.echo_delay, 0)

        def phase_update(phase_reg, i):
            sync()
            phase_reg += float((self.cavity_phases[i] + np.pi/2.0) / np.pi)
            self.update_phase(phase_reg, self.cavity, self.t_mixer_calc)
            sync()

        def control_circuit(i):
            sync()
            phase_reg = FloatRegister()
            phase_reg <<= 0.0
            sync()
            with Repeat(2*self.xp_rounds):
                sbs_step(i)
                reset()
                phase_update(phase_reg, i)
            sync()

        @subroutine
        def reward_circuit(s):
            self.reset_feedback_with_echo(self.echo_delay, self.final_delay, log=True, res_name='m1_'+str(s))
            self.displacement_phase_estimation(s, self.tau_stabilizer, res_name='m2_'+str(s), amp='dynamic')
            delay(self.loop_delay)

        # experience collection loop
        for i in range(self.batch_size):
            for s in self.stabilizers:
                control_circuit(i)
                reward_circuit(s)
