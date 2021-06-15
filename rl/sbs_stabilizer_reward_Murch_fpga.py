# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:21:36 2021

@author: Vladimir Sivak
"""
from init_script import *
import numpy as np
from gkp_exp.gkp_qec.GKP import GKP

class sbs_stabilizer_reward_Murch_fpga(FPGAExperiment, GKP):
    export_txt = False
    save_log = False
    
    """ Autonomous GKP stabilization with SBS protocol and Murch cooling."""
    loop_delay = FloatParameter(4e6)
    batch_size = IntParameter(10)
    opt_file = StringParameter('')
    xp_rounds = IntParameter(15)
    tau_stabilizer = IntParameter(50)
    cal_dir = StringParameter('')
    stabilizers = StringParameter('x+,x-,p+,p-')

    # Fixed parameters of Murch cooling
    cool_duration_ns = IntParameter(100)
    qubit_ramp_ns = IntParameter(200)
    readout_ramp_ns = IntParameter(60)
    final_delay = IntParameter(0)
    

    def sequence(self):
        params = np.load(self.opt_file, allow_pickle=True)
        
        # load SBS pulse sequences from file
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]

        # load Murch cooling parameters from file
        self.Murch_qubit_amp = params['Murch_amp'][:,0]
        self.Murch_readout_amp = params['Murch_amp'][:,1]
        self.Murch_qubit_detune_MHz = params['Murch_detune_MHz'][:,0]
        self.Murch_readout_detune_MHz = params['Murch_detune_MHz'][:,1]
        self.Murch_qubit_phase = params['Murch_phi'][:,0]
        self.Murch_qubit_angle = params['Murch_phi'][:,1]

        # define mode objects
        self.readout, self.qubit, self.cavity = readout, qubit, cavity
        self.qubit_detuned, self.readout_detuned = qubit_ef, readout_aux

        stabilizer_phase_estimation = self.stabilizer_phase_estimation(
                self.tau_stabilizer, self.cal_dir)

        echo_delay, final_delay = 873, 270
        reset_feedback_with_echo = lambda: self.reset_feedback_with_echo(echo_delay, final_delay)

        @subroutine
        def reward_circuit(s):
            reset_feedback_with_echo()
            stabilizer_phase_estimation(s)
            delay(self.loop_delay)

        # experience collection loop
        for i in range(self.batch_size):

            reset = self.reset_autonomous_Murch(self.qubit_detuned, self.readout_detuned,
                    self.cool_duration_ns, self.qubit_ramp_ns, self.readout_ramp_ns,
                    self.Murch_qubit_amp[i], self.Murch_readout_amp[i], 
                    self.Murch_qubit_detune_MHz[i], self.Murch_readout_detune_MHz[i],
                    self.Murch_qubit_angle[i], self.Murch_qubit_phase[i], self.final_delay)

            def sbs_step(s):
                phase = dict(x=0.0, p=np.pi/2.0)
                sync()
                self.cavity.array_pulse(*self.cavity_pulses[i], phase=phase[s])
                self.qubit.array_pulse(*self.qubit_pulses[i])
                sync()

            def control_circuit():
                sync()
                with Repeat(self.xp_rounds):
                    sbs_step('x')
                    reset()
                    sbs_step('p')
                    reset()
                sync()

            for s in self.stabilizers.split(','):
                control_circuit()
                reward_circuit(s)
