# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:42:17 2021

@author: Vladimir Sivak
"""
from init_script import *
import numpy as np

class sbs_Pauli_reward_fpga(FPGAExperiment):
    export_txt = False
    save_log = False

    """ GKP stabilization with SBS protocol. """
    batch_size = IntParameter(10)
    opt_file = StringParameter('')
    rounds = IntParameter(16)

    def sequence(self):
        """
        This sequence has 3 components:
            1) SBS step (ECDC sequence)
            2) qubit reset through feedback with echo pulse
            3) cavity phase update with dynamic mixer
            4) Kerr cancelling drive while updating mixer

        """
        gkp.readout, gkp.qubit, gkp.cavity = readout, qubit, cavity_1

        # load experimental parameters from file
        params = np.load(self.opt_file, allow_pickle=True)
        self.cavity_phases = params['cavity_phases']
        self.Kerr_drive_amps = params['Kerr_drive_amps']
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]
        init_qubit_pulse = (params['init_qubit_pulse'].real, params['init_qubit_pulse'].imag)
        init_cavity_pulse = (params['init_cavity_pulse'].real, params['init_cavity_pulse'].imag)

        # setup qubit mode for Kerr cancelling drive
        self.qubit_detuned = qubit_ef
        self.qubit_detuned.set_detune(gkp.Kerr_drive_detune_MHz*1e6)


        def sbs_step(i):
            sync()
            gkp.cavity.array_pulse(*self.cavity_pulses[i], amp='dynamic')
            gkp.qubit.array_pulse(*self.qubit_pulses[i])
            sync()

        reset = lambda: gkp.reset_feedback_with_echo(gkp.echo_delay, 0)

        def phase_update(phase_reg, i):
            sync()
            self.qubit_detuned.smoothed_constant_pulse(gkp.Kerr_drive_time_ns, 
                        amp=self.Kerr_drive_amps[i], sigma_t=gkp.Kerr_drive_ramp_ns)
            
            # TODO: this mixer update could be done with a subroutine, but 
            # there seem to be some hidden syncs there... 
            phase_reg += float((self.cavity_phases[i] + np.pi/2.0) / np.pi)
            gkp.update_phase(phase_reg, gkp.cavity, gkp.t_mixer_calc_ns)
            sync()
        
        phase_reg = FloatRegister()
        
        def init_setup(phase_reg):
            sync()
            gkp.reset_mixer(gkp.cavity, gkp.t_mixer_calc_ns)
            phase_reg <<= 0.0
            sync()

        def control_circuit(phase_reg, i):
            with Repeat(self.rounds):
                sbs_step(i)
                reset()
                phase_update(phase_reg, i)
            sync()
        
        paulis = {'X' : np.sqrt(np.pi/2.0),
                  'Z' : 1j*np.sqrt(np.pi/2.0)}
        
        @subroutine
        def reward_circuit(s):
            gkp.reset_feedback_with_echo(gkp.echo_delay, gkp.final_delay, log=True, res_name='m1_'+s)
            gkp.displacement_phase_estimation(paulis[s], gkp.t_stabilizer_ns, res_name='m2_'+s, amp='dynamic')
            delay(gkp.loop_delay)

        @subroutine
        def init_circuit(s):
            phase = {'X' : 0.0, 'Z' : np.pi/2.0}
            sync()
            gkp.qubit.array_pulse(*init_qubit_pulse)
            gkp.cavity.array_pulse(*init_cavity_pulse, phase=phase[s])
            sync()

        # experience collection loop
        for i in range(self.batch_size):
            for s in paulis.keys():
                init_setup(phase_reg)
                init_circuit(s)
                control_circuit(phase_reg, i)
                reward_circuit(s)
