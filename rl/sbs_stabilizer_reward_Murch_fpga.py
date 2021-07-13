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
    t_mixer_calc = IntParameter(600)

    # Fixed parameters of Murch cooling
    cool_duration_ns = IntParameter(100)
    qubit_ramp_ns = IntParameter(200)
    readout_ramp_ns = IntParameter(60)
    
    # Parameters for the reward circuit
    echo_delay = IntParameter(868)
    final_delay = IntParameter(140)
    tau_stabilizer = IntParameter(50)
    

    def sequence(self):
        params = np.load(self.opt_file, allow_pickle=True)
        
        # load misc parameters from file
        self.stabilizers = params['stabilizers']
        self.cavity_phases = params['cavity_phases']
        
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

        @subroutine
        def reward_circuit(s):
            self.reset_feedback_with_echo(self.echo_delay, self.final_delay, log=True, res_name='m1_'+str(s))
            self.displacement_phase_estimation(s, self.tau_stabilizer, res_name='m2_'+str(s), amp='dynamic')
            delay(self.loop_delay)

        # experience collection loop
        for i in range(self.batch_size):

            # this has to be done inside the loop because the detunings are set up inside this function
            reset = self.reset_autonomous_Murch(self.qubit_detuned, self.readout_detuned,
                    self.cool_duration_ns, self.qubit_ramp_ns, self.readout_ramp_ns,
                    self.Murch_qubit_amp[i], self.Murch_readout_amp[i], 
                    self.Murch_qubit_detune_MHz[i], self.Murch_readout_detune_MHz[i],
                    self.Murch_qubit_angle[i], self.Murch_qubit_phase[i], 0)
    
            def sbs_step():
                sync()
                self.cavity.array_pulse(*self.cavity_pulses[i], amp='dynamic')
                self.qubit.array_pulse(*self.qubit_pulses[i])
                sync()
    
            def phase_update(phase_reg):
                sync()
    #            self.qubit_detuned.smoothed_constant_pulse(self.Kerr_drive_time_ns, 
    #                        amp=self.Kerr_drive_amps[i], sigma_t=self.Kerr_drive_ramp_ns)
                
                # TODO: this mixer update could be done with a subroutine, but 
                # there seem to be some hidden syncs there... 
                phase_reg += float((self.cavity_phases[i] + np.pi/2.0) / np.pi)
                c = FloatRegister()
                s = FloatRegister()
                c = af_cos(phase_reg)
                s = af_sin(phase_reg)
                DynamicMixer[0][0] <<= c
                DynamicMixer[1][0] <<= s
                DynamicMixer[0][1] <<= -s
                DynamicMixer[1][1] <<= c
                self.cavity.delay(self.t_mixer_calc)
                self.cavity.load_mixer()
                sync()
    
            def control_circuit():
                sync()
                phase_reg = FloatRegister()
                phase_reg <<= 0.0
                sync()
                with Repeat(2*self.xp_rounds):
                    sbs_step()
                    reset()
                    phase_update(phase_reg)
                sync()
    

            for s in self.stabilizers:
                control_circuit()
                reward_circuit(s)
