# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:32:46 2021

@author: Vladimir Sivak
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP

class sbs_stabilizer_reward_jumps_fpga(FPGAExperiment, GKP):
    export_txt = False
    save_log = False
    
    """ GKP stabilization with SBS protocol. """
    loop_delay = FloatParameter(4e6)
    batch_size = IntParameter(10)
    opt_file = StringParameter('')
    xp_rounds = IntParameter(15)
    tau_stabilizer = IntParameter(50)
    cal_dir = StringParameter(r'D:\DATA\exp\2021-05-13_cooldown\CD_fixed_time_amp_cal')
    echo_delay = IntParameter(880)
    final_delay = IntParameter(92)

    def sequence(self):
        self.readout, self.qubit, self.cavity = readout, qubit, cavity
        # load SBS pulse sequences from file
        params = np.load(self.opt_file, allow_pickle=True)
        self.stabilizers = params['stabilizers']
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]

        reset = lambda: self.reset_feedback_with_echo(self.echo_delay, self.final_delay)

        def sbs_step(i, s):
            phase = dict(x=0.0, p=np.pi/2.0)
            sync()
            self.cavity.array_pulse(*self.cavity_pulses[i], phase=phase[s])
            self.qubit.array_pulse(*self.qubit_pulses[i])
            sync()

        def control_circuit(i, s):
            sbs_step(i, s)
            reset()

        @subroutine
        def reward_circuit(s):
            self.reset_feedback_with_echo(self.echo_delay, self.final_delay, log=True, res_name='m1_'+str(s))
            self.displacement_phase_estimation(s, self.tau_stabilizer, self.cal_dir, res_name='m2_'+str(s))
            delay(self.loop_delay)
        
        @subroutine
        def init_circuit():
            pass

        R0 = Register(0)
        R1 = Register(2*self.xp_rounds)
        C = Register(0)
        
        # experience collection loop
        for i in range(self.batch_size):
            for j, s in enumerate(self.stabilizers):
                sync()
                loop_label = str(i) + '_' + str(j) + '_'
                
                R0 <<= 0
                R1 <<= 2*self.xp_rounds
                C  <<= 0

                label_next('stabilization'+loop_label)

                label_next('x_round'+loop_label)
                if_then(C==R0, 'init'+loop_label)
                if_then(C==R1, 'reward'+loop_label)
                control_circuit(i, 'x')
                C += 1
                goto('p_round'+loop_label)

                label_next('p_round'+loop_label)
                if_then(C==R0, 'init'+loop_label)
                if_then(C==R1, 'reward'+loop_label)
                control_circuit(i, 'p')
                C += 1
                goto('x_round'+loop_label)

                label_next('init'+loop_label)
                init_circuit()
                C += 1
                goto('stabilization'+loop_label)

                label_next('reward'+loop_label)
                reward_circuit(s)
