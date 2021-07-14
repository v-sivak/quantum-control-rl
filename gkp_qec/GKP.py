# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:44:39 2021

@author: qulab
"""
import numpy as np
from fpga_lib.dsl import *
from fpga_lib.parameters import *
from fpga_lib.experiments import *
from fpga_lib import entities

from gkp_exp.CD_gate.conditional_displacement_compiler import SBS_simple_compiler, ConditionalDisplacementCompiler, ECD_control_simple_compiler


class GKP(Calibratable):
    """
    Args:
        cal_dir (str): directory with CD gate amplitude calibrations
    """
    # Params for ECDC sequences
    qubit_pulse_pad = IntParameter(4)
    s_tau_ns = IntParameter(20)
    b_tau_ns = IntParameter(150)
    cal_dir = StringParameter(r'D:\DATA\exp\2021-06-28_cooldown\CD_fixed_time_amp_cal')

    # Params for echoed feedback reset
    echo_delay = IntParameter(868)
    final_delay = IntParameter(64)
    
    # Params for Kerr-cancelling drive
    Kerr_drive_time_ns = IntParameter(200)
    Kerr_drive_ramp_ns = IntParameter(200)
    Kerr_drive_detune_MHz = FloatParameter(15)
    
    # Params misc
    loop_delay = IntParameter(4e6)
    t_stabilizer_ns = IntParameter(150)
    t_mixer_calc_ns = IntParameter(600)
    
    def __init__(self, name='gkp'):
        super(GKP, self).__init__(name)
    
    @subroutine
    def reset_feedback_with_echo(self, echo_delay, final_delay, feedback_delay=0, log=False, res_name='default'):
        """
        Feedback reset with echo during readout.
        
        Args:
            echo_delay (int): delay in [ns] from the beginning of the readout
                to the qubit echo pulse.
            final_delay (int): delay in [ns] after the feedback to cancel 
                deterministic (state-independent) cavity rotation.
            feedback_delay (int): delay in [ns] of the feedback pulse. There 
                will be additional processing time contribution on top of this.
            log (bool): flag to log the measurement outcome.
            res_name (str): name of the result if measurement is logged.
        """
        sync()
        delay(echo_delay, channel=self.qubit.chan, round=True)
        self.qubit.flip() # echo pulse
        self.readout(wait_result=True, log=log, sync_at_beginning=False, **{res_name:'se'})
        sync()
        delay(feedback_delay, round=True)
        if_then_else(self.qubit.measured_low(), 'flip', 'wait')
        label_next('flip')
        self.qubit.flip()
        goto('continue')
        label_next('wait')
        delay(self.qubit.pulse.length)
        label_next('continue')
        delay(final_delay, round=True)
        sync()


    def reset_autonomous_Murch(self, qubit_detuned_obj, readout_detuned_obj,
                    cool_duration_ns, qubit_ramp_ns, readout_ramp_ns,
                    qubit_amp, readout_amp, qubit_detune_MHz, readout_detune_MHz,
                    qubit_angle, qubit_phase, final_delay):
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
            qubit_angle, qubit_phase (float): final qubit rotation parameters
            final_delay (int): delay in [ns] after the cooling protocol
            
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
            self.qubit.rotate(qubit_angle, qubit_phase)
            sync()
            delay(final_delay, round=True)

        return lambda: cooling_Murch()


    def sbs(self, eps1, eps2, beta, s_tau_ns, b_tau_ns):
        """
        Single step of SBS protocol based on this Baptiste paper:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.260509
        
        The pulse sequence is compile based on the independent calibration of
        the conditional displacement amplitude. 
        
        Args:
            eps1, eps2 (float): 1st/2nd small CD amplitude
            beta (float): big CD amplitude
            
            s_tau_ns, b_tau_ns (int): wait time in the small/big CD gate
        """
        CD_compiler_kwargs = dict(qubit_pulse_pad=self.qubit_pulse_pad)
        C = SBS_simple_compiler(CD_compiler_kwargs, self.cal_dir)
        
        cavity_pulse, qubit_pulse = C.make_pulse(1j*eps1/2.0, 1j*eps2/2.0, beta,
                                                 s_tau_ns, b_tau_ns)

        def sbs_step(s):
            """
            Args:
                s (str): stabilization qudrature, either 'x' or 'p' 
            """
            phase = dict(x=0.0, p=np.pi/2.0)
            sync()
            self.cavity.array_pulse(*cavity_pulse, phase=phase[s])
            self.qubit.array_pulse(*qubit_pulse)
            sync()
        
        return sbs_step

    def load_sbs_sequence(self, s_tau, b_tau, ECD_filename, version):
        """
        Args:
            version (str): 
                - v1 is a simple version in which only ECD parameters beta & phi
                  are loaded from the file.
                - v2 is a more complicated version in which qubit detunings and
                  parameters of the pi-pulses are also used in addition to beta & phi.
                - v3 is like v2 but it returns an sbs_step that uses dynamix mixer.
        """
        if version == 'v1':
            data = np.load(ECD_filename, allow_pickle=True)
            beta, phi = data['beta'], data['phi']
            tau = np.array([s_tau, b_tau, s_tau, 0])
    
            CD_compiler_kwargs = dict(qubit_pulse_pad=self.qubit_pulse_pad)
            C = ECD_control_simple_compiler(CD_compiler_kwargs, self.cal_dir)
            c_pulse, q_pulse = C.make_pulse(beta, phi, tau)
        if version in ['v2', 'v3']:
            data = np.load(ECD_filename, allow_pickle=True)
            beta, phi, phi_CD, detune = data['beta'], data['phi'], data['flip'], data['detune']
            tau = np.array([s_tau, b_tau, s_tau, 0])
    
            CD_compiler_kwargs = dict(qubit_pulse_pad=self.qubit_pulse_pad)
            C = ECD_control_simple_compiler(CD_compiler_kwargs, self.cal_dir)
            c_pulse, q_pulse = C.make_pulse_v2(beta, phi, phi_CD, tau, detune)
        
        if version in ['v1', 'v2']:
            def sbs_step(s):
                """
                Args:
                    s (str): stabilizer direction, either 'x' or 'p'
                """
                phase = dict(x=0.0, p=np.pi/2.0)
                sync()
                self.cavity.array_pulse(c_pulse.real, c_pulse.imag, phase=phase[s])
                self.qubit.array_pulse(q_pulse.real, q_pulse.imag)
                sync()
        
        if version == 'v3':
            def sbs_step():
                sync()
                self.cavity.array_pulse(c_pulse.real, c_pulse.imag, amp='dynamic')
                self.qubit.array_pulse(q_pulse.real, q_pulse.imag)
                sync()            
        
        return sbs_step
        

    def snap(self, snap_length):
        delay(snap_length)
        

    def stabilizer_phase_estimation(self, tau_ns):
        
        beta = np.sqrt(2*np.pi) # stabilizer CD amplitude
        C = ConditionalDisplacementCompiler(qubit_pulse_pad=self.qubit_pulse_pad)
        CD_params = C.CD_params_fixed_tau_from_cal(beta, tau_ns, self.cal_dir)
        cavity_pulse, qubit_pulse = C.make_pulse(*CD_params)
        
        def stabilizer_phase_estimation(s):
            phase = {'x' : 0.0, 'x+' : 0.0, 'x-' : np.pi, 
                     'p' : np.pi/2.0, 'p+' : np.pi/2.0, 'p-' : -np.pi/2.0}
            sync()
            self.qubit.pi2_pulse(phase=np.pi/2.0)
            sync()
            self.cavity.array_pulse(*cavity_pulse, phase=phase[s])
            self.qubit.array_pulse(*qubit_pulse)
            sync()
            self.qubit.pi2_pulse(phase=-np.pi/2.0)
            sync()
            delay(24)
            self.readout(**{s:'se'})
            sync()
        
        return stabilizer_phase_estimation
        
        

    def displacement_phase_estimation(self, beta, tau_ns, res_name, 
                                      echo_params=None, amp=1):
        
        C = ConditionalDisplacementCompiler(qubit_pulse_pad=self.qubit_pulse_pad)
        CD_params = C.CD_params_fixed_tau_from_cal(beta, tau_ns, self.cal_dir)
        cavity_pulse, qubit_pulse = C.make_pulse(*CD_params)
        
        sync()
        self.qubit.pi2_pulse(phase=np.pi/2.0)
        sync()
        self.cavity.array_pulse(*cavity_pulse, amp=amp)
        self.qubit.array_pulse(*qubit_pulse)
        sync()
        self.qubit.pi2_pulse(phase=-np.pi/2.0)
        sync()
        delay(24)
        if echo_params is not None:
            self.reset_feedback_with_echo(
                    echo_params['echo_delay'], echo_params['final_delay'], 
                    log=True, res_name=res_name)
        else:
            self.readout(**{res_name:'se'})
        sync()
        
    @subroutine
    def update_phase(self, phase_reg, mode, t_mixer_calc=400):
        c = FloatRegister()
        s = FloatRegister()
        c = af_cos(phase_reg)
        s = af_sin(phase_reg)
        DynamicMixer[0][0] <<= c
        DynamicMixer[1][0] <<= s
        DynamicMixer[0][1] <<= -s
        DynamicMixer[1][1] <<= c
        mode.delay(t_mixer_calc)
        mode.load_mixer()