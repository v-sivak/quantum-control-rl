# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:42:17 2021

@author: Vladimir Sivak
"""
from init_script import *
import numpy as np

class sbs_Pauli_reward_fpga_v2(FPGAExperiment):
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
            2) qubit reset through feedback
            3) cavity phase update with dynamic mixer
            4) Kerr cancelling drive while updating mixer

        """
        gkp.readout, gkp.qubit, gkp.cavity, gkp.qubit_detuned = readout, qubit, cavity_1, qubit_ef

        # load experimental parameters from file
        params = np.load(self.opt_file, allow_pickle=True)
        self.cavity_phases = params['cavity_phases']
        Kerr_g_drive_amps = params['Kerr_drive_amps'][:,0]
        Kerr_e_drive_amps = params['Kerr_drive_amps'][:,1]
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]
        init_qubit_pulse = (params['init_qubit_pulse'].real, params['init_qubit_pulse'].imag)
        init_cavity_pulse = (params['init_cavity_pulse'].real, params['init_cavity_pulse'].imag)
        phase_g, phase_e = params['cavity_phases'][:,0], params['cavity_phases'][:,1]
        qb_detune, qb_drag = params['qb_detune'], params['qb_drag']

        def sbs_step(i):
            sync()
            gkp.cavity.array_pulse(*self.cavity_pulses[i], amp='dynamic')
            gkp.qubit.array_pulse(*self.qubit_pulses[i])
            sync()

        def reset(phase_reg, phase_g_reg, phase_e_reg, i):
            gkp.reset_feedback_with_phase_update_and_Kerr_drive(
                    phase_reg, phase_g_reg, phase_e_reg, detune=qb_detune[i], drag=qb_drag[i],
                    Kerr_g_amp=Kerr_g_drive_amps[i], Kerr_e_amp=Kerr_e_drive_amps[i])

        def init_setup(phase_reg, phase_g_reg, phase_e_reg, i):
            sync()
            
            sync()
            phase_g_reg <<= (float(phase_g[i]) + np.pi/2.0) / np.pi
            phase_e_reg <<= (float(phase_e[i]) + np.pi/2.0) / np.pi
            phase_reg <<= 0.0
            sync()
            gkp.reset_mixer(gkp.cavity, gkp.t_mixer_calc_ns)
            sync()

        def control_circuit(phase_reg, phase_g_reg, phase_e_reg, i):
            with Repeat(self.rounds):
                sbs_step(i)
                reset(phase_reg, phase_g_reg, phase_e_reg, i)
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

        # Some initial setup
        gkp.qubit_detuned.set_detune(gkp.Kerr_drive_detune_MHz*1e6)

        phase_reg = FloatRegister()
        phase_g_reg = FloatRegister()
        phase_e_reg = FloatRegister()

        # experience collection loop
        for i in range(self.batch_size):
            for s in paulis.keys():
                init_setup(phase_reg, phase_g_reg, phase_e_reg, i)
                init_circuit(s)
                control_circuit(phase_reg, phase_g_reg, phase_e_reg, i)
                reward_circuit(s)
