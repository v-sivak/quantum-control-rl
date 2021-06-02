# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:44:39 2021

@author: qulab
"""

from init_script import *
from gkp_exp.CD_gate.conditional_displacement_compiler import SBS_simple_compiler, ConditionalDisplacementCompiler
import numpy as np


class GKP():

    @subroutine
    def reset_feedback_with_echo(self, echo_delay, final_delay):
        """
        Feedback reset with echo during readout.
        
        Args:
            echo_delay (int): delay in [ns] from the beginning of the readout
                to the qubit echo pulse.
            final_delay (int): delay in [ns] after the feedback to cancel 
                deterministic (state-independent) cavity rotation.
        """
        sync()
        delay(echo_delay, channel=self.qubit.chan)
        self.qubit.flip() # echo pulse
        readout(wait_result=True, log=False, sync_at_beginning=False)
        sync()
        if_then_else(self.qubit.measured_low(), 'flip', 'wait')
        label_next('flip')
        self.qubit.flip()
        goto('continue')
        label_next('wait')
        delay(self.qubit.pulse.length)
        label_next('continue')
        delay(final_delay)
        sync()


    def reset_autonomous_Murch(self, qubit_detuned_obj, readout_detuned_obj,
                    cool_duration_ns, qubit_ramp_ns, readout_ramp_ns,
                    qubit_amp, readout_amp, qubit_detune_MHz, readout_detune_MHz):
        """
        Setup autonomous qubit cooling based on this Murch paper:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.183602
        
        Args:
            qubit_detuned_obj (Mode): qubit mode to use in the protocol
            readout_detuned_obj (Mode): readout mode to use in the protocol
            cool_duration_ns (int): how long in [ns] to hold the constant
                Rabi drive on the qubit after ramping it up.
            qubit_ramp_ns (int): duration in [ns] of the qubit Rabi drive
                ramp up/down.
            readout_ramp_ns (int): duration in [ns] of the detuned readout
                drive ramp up/down. This can typically be shorter than the
                qubit ramp because the pulse is far detuned.
            qubit_amp (float): amplitude of the qubit Rabi pulse.
            readout_amp (float): amplitude of the detuned readout pulse.
            readout_detune_MHz (float): detuning of the readout pulse in [MHz]. 
                Ideally equal to the qubit Rabi rate.
            qubit_detune_MHz (float): detuning of the qubit pulse in [MHz]
        
        Returns:
            cooling subroutine.
        """
        self.qubit_detuned = qubit_detuned_obj
        self.readout_detuned = readout_detuned_obj

        sync()
        self.readout_detuned.set_detune(readout_detune_MHz*1e6)
        self.qubit_detuned.set_detune(qubit_detune_MHz*1e6)
        sync()
        
        qubit_pump_time = cool_duration_ns
        readout_pump_time = cool_duration_ns+2*qubit_ramp_ns-2*readout_ramp_ns
        
        @subroutine
        def cooling_Murch():
            sync()
            self.readout_detuned.smoothed_constant_pulse(
                    readout_pump_time, amp=readout_amp, sigma_t=readout_ramp_ns)
            self.qubit_detuned.smoothed_constant_pulse(
                    qubit_pump_time, amp=qubit_amp, sigma_t=qubit_ramp_ns)
            sync()

        return lambda: cooling_Murch()


    def sbs(self, eps1, eps2, beta, s_tau_ns, b_tau_ns, cal_dir):
        """
        Single step of SBS protocol based on this Baptiste paper:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.260509
        
        The pulse sequence is compile based on the independent calibration of
        the conditional displacement amplitude. 
        
        Args:
            eps1, eps2 (float): 1st/2nd small CD amplitude
            beta (float): big CD amplitude
            
            s_tau_ns, b_tau_ns (int): wait time in the small/big CD gate
            cal_dir (str): directory with CD gate calibrations
        """
        CD_compiler_kwargs = dict(qubit_pulse_pad=0)
        C = SBS_simple_compiler(CD_compiler_kwargs, cal_dir)
        
        cavity_pulse, qubit_pulse = C.make_pulse(1j*eps1/2.0, 1j*eps2/2.0, beta,
                                                 s_tau_ns, b_tau_ns)

        qubit_pulse_sbs = qubit_pulse
        cavity_pulse_sbs = {'x': (cavity_pulse[0], cavity_pulse[1]),
                            'p': (-cavity_pulse[1], cavity_pulse[0])}

        def sbs_step(s):
            """
            Args:
                s (str): stabilization qudrature, either 'x' or 'p' 
            """
            sync()
            self.cavity.array_pulse(*cavity_pulse_sbs[s])
            self.qubit.array_pulse(*qubit_pulse_sbs)
            sync()
        
        return sbs_step


    def snap(self, snap_length):
        delay(snap_length)
        

    def stabilizer_phase_estimation(self, tau_ns, cal_dir):
        
        C = ConditionalDisplacementCompiler()
        CD_params = C.CD_params_fixed_tau_from_cal(
                np.sqrt(2*np.pi), tau_ns, cal_dir)
        cavity_pulse, qubit_pulse = C.make_pulse(*CD_params)

        qubit_stabilizer_CD_pulse = qubit_pulse
        cavity_stabilizer_CD_pulse = {'x': (cavity_pulse[0], cavity_pulse[1]), 
                                      'p': (-cavity_pulse[1], cavity_pulse[0])}
        
        def stabilizer_phase_estimation(s):
            sync()
            self.qubit.pi2_pulse(phase=np.pi/2.0)
            sync()
            self.cavity.array_pulse(*cavity_stabilizer_CD_pulse[s])
            self.qubit.array_pulse(*qubit_stabilizer_CD_pulse)
            sync()
            self.qubit.pi2_pulse(phase=np.pi/2.0)
            sync()
            delay(24)
            readout(**{s:'se'})
            sync()
        
        return stabilizer_phase_estimation
        
        
        