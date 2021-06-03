# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:53:11 2021

@author: qulab
"""
import os
import numpy as np
import math
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import CubicSpline
from init_script import cavity, qubit
from ECD_control.ECD_pulse_construction.ECD_pulse_construction import FakeStorage, conditional_displacement_circuit

def get_calibrated_pulse(pulse, zero_pad=None, detune=0):
    """ 
    Get calibrated pulse with correct dac amplitude, detuning etc.
    For qubit this is a pi-pulse, for cavity this is a unit displacement.
    
    If the pulse is shorter than 24 ns, fpga_lib will pad it to 24, and
    this function strips that padding and optionally adds its own padding.
    """
    i, q = pulse.make_wave(zero_pad=False)
    f = pulse.detune + detune
    t_offset = (24 - len(i)) / 2 if len(i)<24 else 0
    t = (np.arange(len(i)) + t_offset)*1e-9 # convert to ns
    i_prime = np.cos(2*np.pi*f*t)*i + np.sin(2*np.pi*f*t)*q
    q_prime = np.cos(2*np.pi*f*t)*q - np.sin(2*np.pi*f*t)*i
    A_complex = pulse.unit_amp * (i_prime + 1j * q_prime)
    if zero_pad is not None:
        A_complex = np.pad(A_complex, zero_pad, 'constant')
    return A_complex

class ConditionalDisplacementCompiler():
    """
    This uses the definition CD(beta) = D(sigma_z*beta/2).
    """
    def __init__(self, qubit_pulse_shift=0, qubit_pulse_pad=0,
                 pad_clock_cycle=True):
        """
        Args:
            qubit_pulse_shift (int): by how much to advance or delay the qubit 
                pulse relative to the cavity pulse (in nanoseconds).
            qubit_pulse_pad (int): by how much to pad the qubit pulse on each
                side with zeros (in nanoseconds).
            pad_clock_cycle (bool): flag to pad the gate to be multiple of 4 ns
        """
        self.qubit_pulse_shift = qubit_pulse_shift
        self.qubit_pulse_pad = qubit_pulse_pad
        self.pad_clock_cycle = pad_clock_cycle
    
    # TODO: update to match make_pulse arguments
    def CD_params_fixed_tau(self, beta, tau_ns):
        """
        Find parameters for CD gate based on simple constant chi model.
        The displacements can be complex-valued here.
        Angles are taken in the counter-clockwise direction, so normally
        phi_e will be positive and phi_g will be negative.
        """
        alpha = beta / 2. / np.sin(2*np.pi*cavity.chi*tau_ns*1e-9) * 1j
        if np.abs(alpha)*cavity.displace.unit_amp > 0.95:
            alpha *= (0.95/cavity.displace.unit_amp) / np.abs(alpha)
        phi_g = np.zeros_like(alpha)
        phi_e = np.zeros_like(alpha)
        return (tau_ns, alpha, phi_g, phi_e)

    # TODO: update to match make_pulse arguments
    def CD_params_fixed_alpha(self, beta, alpha_abs):
        tau = 1. / (2*np.pi*cavity.chi) * np.arcsin(np.abs(beta/alpha_abs/2.))
        tau_ns = int(math.ceil(tau*1e9))
        alpha_abs = np.abs(beta) / 2. / np.sin(2*np.pi*cavity.chi*tau_ns*1e-9)
        alpha_phase = np.angle(beta) + np.pi/2
        alpha = alpha_abs * np.exp(1j*alpha_phase)
        phi_g = np.zeros_like(alpha)
        phi_e = np.zeros_like(alpha)
        return (tau_ns, alpha, phi_g, phi_e)

    def CD_params_fixed_tau_from_cal(self, beta, tau_ns, cal_dir):
        """
        Find parameters for CD gate based on a simple calibration.
        Calibration gives a linear fit "alpha = a * beta + b" for a fixed tau,
        this can be obtained from 'CD_fixed_time_amp_cal' experiment.
        The displacements can be complex-valued here.
        Angles are taken in the counter-clockwise direction, so normally
        phi_e will be positive and phi_g will be negative.
        """
        tau_dir = os.path.join(cal_dir, 'tau='+str(int(tau_ns))+'ns')
        data = np.load(os.path.join(tau_dir, 'quadratic_fit.npz'))
        a, b, c = data['a'], data['b'], data['c']
        assert tau_ns == data['tau_ns']
        alpha_abs = a + b * np.abs(beta) + c * np.abs(beta)**2
        alpha_abs_max = 0.95 / cavity.displace.unit_amp
        alpha_abs = np.min([alpha_abs, alpha_abs_max])
        alpha = alpha_abs * np.exp(1j*(np.pi/2.0 + np.angle(beta)))
        CD_params = (alpha, -alpha, -alpha, alpha, tau_ns, tau_ns, 0, np.pi, 0)
        return CD_params


    def make_pulse(self, alpha1, alpha2, alpha3, alpha4, 
                    tau1, tau2, phi, theta, Delta):
        """ Build cavity and qubit sequences for CD gate.

        Args: 
            alpha1-alpha4 (complex): cavity displacement amplitudes
            tau1-tau2 (int): wait times for cavity rotation
            phi, theta (float): phase and angle of qubit rotation
            Delta (float): qubit pulse detuning in MHz

        In the ideal case, we would have the following settings:
            alpha1 = -alpha2 = -alpha3 = alpha4 = alpha
            tau1 = tau2 = tau
            phi, theta = [0, pi]
            Delta = 0
        """
        pi_pulse = get_calibrated_pulse(qubit.pulse, 
                                zero_pad=self.qubit_pulse_pad, detune=Delta)
        D_complex = get_calibrated_pulse(cavity.displace)

        #----- first build cavity pulse ------------
        C_pulse = D_complex * alpha1
        C_pulse = np.concatenate([C_pulse, np.zeros(int(round(tau1)))])
        C_pulse = np.concatenate([C_pulse, D_complex * alpha2])
        C_pulse = np.concatenate([C_pulse, np.zeros(len(pi_pulse))])
        C_pulse = np.concatenate([C_pulse, D_complex * alpha3])
        C_pulse = np.concatenate([C_pulse, np.zeros(int(round(tau2)))])
        C_pulse = np.concatenate([C_pulse, D_complex * alpha4])

        #----- now build qubit pulse ------------
        Q_complex = pi_pulse * theta / np.pi * np.exp(1j*phi)
        flip_delay = (len(C_pulse) - len(pi_pulse)) / 2
        Q_pulse = np.concatenate([np.zeros(flip_delay), Q_complex, np.zeros(flip_delay)])
        
        # shift the qubit pulse to compensate electrical delay
        Q_pulse = np.roll(Q_pulse, self.qubit_pulse_shift)

        # make sure pulse length is multiple of 4 ns
        if self.pad_clock_cycle:
            zero_pad = np.zeros(4 - (len(C_pulse) % 4))
            C_pulse = np.concatenate([C_pulse, zero_pad])
            Q_pulse = np.concatenate([Q_pulse, zero_pad])

        return (C_pulse.real, C_pulse.imag), (Q_pulse.real, Q_pulse.imag)


class ECD_control_simple_compiler():
    
    def __init__(self, CD_compiler_kwargs, cal_dir):
        """
        Args:
            CD_compiler_kwargs (dict): to be used on initialization of the
                'ConditionalDisplacementCompiler' instance.
            cal_dir (str): directory with CD gate calibrations
        """
        CD_compiler_kwargs['pad_clock_cycle'] = False
        self.CD = ConditionalDisplacementCompiler(**CD_compiler_kwargs)
        self.pi_pulse = get_calibrated_pulse(qubit.pulse, zero_pad=self.CD.qubit_pulse_pad)
        
        func = self.CD.CD_params_fixed_tau_from_cal
        self.CD_params_func = lambda beta, tau: func(beta, tau, cal_dir)

    def make_pulse(self, beta, phi, tau):
        """
        Args:
            beta (array([T,2]), flaot32)
            phi  (array([T,2]), float32)
            tau  (array(T), float32)
        """
        T = beta.shape[0] # protocol duration (number of steps)
        C_pulse, Q_pulse = np.array([]), np.array([])
        
        for t in range(T):
            # First create the qubit rotation gate
            phase_t, angle_t = phi[t,0], phi[t,1]
            qb_rotation = self.pi_pulse * angle_t / np.pi * np.exp(1j*phase_t)
            C_pulse = np.concatenate([C_pulse, np.zeros_like(qb_rotation)])
            Q_pulse = np.concatenate([Q_pulse, qb_rotation])
            
            # Then create the CD gate
            beta_t = beta[t,0] + 1j*beta[t,1]
            tau_t = tau[t]
            if t < T-1:
                CD_params = self.CD_params_func(beta_t, tau_t)
                cav_CD, qb_CD = self.CD.make_pulse(*CD_params)
                C_pulse = np.concatenate([C_pulse, cav_CD[0] + 1j*cav_CD[1]])
                Q_pulse = np.concatenate([Q_pulse, qb_CD[0] + 1j*qb_CD[1]])
            elif t == T-1:
                # Implement last CD as simple displacement
                D = get_calibrated_pulse(cavity.displace)
                C_pulse = np.concatenate([C_pulse, D * beta_t/2.0])
                Q_pulse = np.concatenate([Q_pulse, np.zeros_like(D)])
        
        # make sure total pulse length is multiple of 4 ns
        zero_pad = np.zeros(4 - (len(C_pulse) % 4))
        C_pulse = np.concatenate([C_pulse, zero_pad])
        Q_pulse = np.concatenate([Q_pulse, zero_pad])
            
        return C_pulse, Q_pulse



class SBS_simple_compiler():
    
    def __init__(self, CD_compiler_kwargs, cal_dir):
        """ Super primitive compiler for the SBS sequence, see this paper:
            https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.125.260509
        See docs for ECD_control_simple_compiler. """
        self.C = ECD_control_simple_compiler(CD_compiler_kwargs, cal_dir)
    
    def make_pulse(self, eps1, eps2, beta, s_tau, b_tau):
        # conditional displacement amplitudes and wait times
        betas_complex = np.array([eps1, beta, eps2, 0])
        betas = np.stack([betas_complex.real, betas_complex.imag], axis=1)
        taus = np.array([s_tau, b_tau, s_tau, 0])
        
        # qubit rotation phases and angles
        phases = np.array([np.pi/2., 0., 0., -np.pi/2.])
        angles = np.array([np.pi/2., -np.pi/2., np.pi/2., np.pi/2.])
        phis = np.stack([phases, angles], axis=1)
        
        C_pulse, Q_pulse = self.C.make_pulse(betas, phis, taus)        
        return (C_pulse.real, C_pulse.imag), (Q_pulse.real, Q_pulse.imag)


    
class ECD_control_Alec_compiler():
    
    
    def __init__(self, alpha_CD, buffer_time):
        """
        Args:
            alpha_CD (float): the alpha used during the ECD gates.
            buffer_time (float): buffer time between R and ECD pulses.
        """
        epsilon_m = 1e9 / np.trapz(cavity.displace.make_wave()[0])
        
        # naming conventions, sign conventions and factors of 2 are different
        storage_params = {'chi_kHz': -cavity.chi*1e-3, 
                          'chi_prime_Hz': -cavity.chi_prime/2.0, 
                          'Ks_Hz': cavity.kerr/2.0, 
                          'epsilon_m_MHz': epsilon_m*1e-6, 
                          'unit_amp': cavity.displace.unit_amp, 
                          'sigma': cavity.displace.sigma, 
                          'chop': cavity.displace.chop}
        self.storage = FakeStorage(**storage_params)
        self.qubit = qubit
        
        self.alpha = alpha_CD
        self.buffer_time = buffer_time
      
    def make_pulse(self, beta, phi):
        """
        Args:
            beta (array([T,2], flaot32)): Re and Im part of the displacement
            phi  (array([T,2], float32)): phase and angle of qubit rotation
        """
        betas = beta[:,0] + 1j*beta[:,1]
        phis, thetas = phi[:,0], phi[:,1]
        print(betas)
        cd_circuit_dict = conditional_displacement_circuit(betas, phis, thetas, 
                    self.storage, self.qubit, self.alpha,  buffer_time=self.buffer_time, 
                    kerr_correction=False, chi_prime_correction=True, final_disp=False, pad=True)
        
        cavity_dac_pulse, qubit_dac_pulse, = cd_circuit_dict['cavity_dac_pulse'], cd_circuit_dict['qubit_dac_pulse']
        return cavity_dac_pulse, qubit_dac_pulse
    



class SBS_Alec_compiler():
    def __init__(self, ECD_control_kwargs={}):
        self.C = ECD_control_Alec_compiler(**ECD_control_kwargs)
    
    def make_pulse(self, eps1, eps2, beta):
        
        betas = np.array([eps1, beta, eps2, 0])
        phis = np.array([np.pi/2, 0, 0, -np.pi/2])
        thetas = np.array([np.pi/2, -np.pi/2, np.pi/2, np.pi/2])
        
        beta_sbs = np.stack([betas.real, betas.imag], axis=1)
        phi_sbs =  np.stack([phis, thetas], axis=1)

        C_pulse, Q_pulse = self.C.make_pulse(beta_sbs, phi_sbs)
        return (C_pulse.real, C_pulse.imag), (Q_pulse.real, Q_pulse.imag)
    
    
