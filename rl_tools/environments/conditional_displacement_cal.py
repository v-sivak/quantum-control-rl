# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:39:39 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from math import pi, sqrt
from tensorflow import complex64 as c64
from gkp.tf_env.tf_env import GKP
from tf_agents import specs
from simulator.hilbert_spaces import OscillatorQubit

class PhaseSpaceSimulator():
    """
    This class is for semiclassical simulation of the phase space evolution.
    It's used to avoid simulating a large Hilbert space of the oscillator, 
    and instead perform simple calculation to approximately find the result
    of the sequence of large displacements and delays, implementing the CD.
    
    We assume that the state at every iteration is represented as
    |psi_k> = e^{i*phase_k} * e^{-i*angle_k*n} * D(alpha_k) |psi_0>
    and this class only updates the parameters (phase_k, angle_k, alpha_k)
    after each gate.
    
    The update rules are:
        
    1) Rotation by angle 'F':
        phase <- phase
        angle <- angle + F
        alpha <- alpha
        
    2) Displacement with amplitude 'A':
        phase <- phase + Im(conj(alpha) * A * e^{i*angle})
        angle <- angle
        alpha <- alpha + A * e^{i*angle}
        
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.reset()

        self.chi = 184e3
        
        # load experimental data and use interpolation to define rotation
        # frequency at any nbar in the cavity.
        nbar = np.load(r'Z:\tmp\for Vlad\from_vlad\nbar.npy').astype(np.float32)
        rotation_freq = dict(
            g = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_g.npy').astype(np.float32),
            e = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_e.npy').astype(np.float32))
        
        self.freq = lambda s, n: tf.cast(tfp.math.interp_regular_1d_grid(
            n, nbar.min(), nbar.max(), rotation_freq[s]), tf.float32)

    
    def reset(self):
        zeros = tf.zeros(self.batch_size)
        zeros_c = tf.cast(zeros, c64)
        self.phase = dict(g=zeros, e=zeros)
        self.alpha = dict(g=zeros_c, e=zeros_c)
        self.angle = dict(g=zeros, e=zeros)
    
    # def freq(self, s, nbar):
    #     self.chi = 184e3
    #     sign = 1 if s=='g' else -1
    #     return 1/2*sign*self.chi
        
    def nbar(self, s):
        return tf.math.abs(self.alpha[s])**2
    
    def rotate(self, s, phi):
        """Rotate oscillator phase space projection onto qubit s-state."""
        self.angle[s] += phi

    def delay(self, tau):
        """Delay by time 'tau' rotates the phase space by different angles 
        for qubit 'g' and 'e' states."""
        for s in ['g','e']:
            phi = 2*pi * self.freq(s, self.nbar(s)) * tau 
            self.rotate(s, phi)

    def displace(self, alpha):
        """Displacement by (typically large) amplitude 'alpha'."""
        for s in ['g','e']:
            delta_alpha = alpha * tf.math.exp(1j*tf.cast(self.angle[s], c64))
            delta_phase = tf.math.imag(delta_alpha*tf.math.conj(self.alpha[s]))
            self.alpha[s] += delta_alpha
            self.phase[s] += delta_phase

    def flip(self):
        """Flip the assignment of the qubit states (i.e. qubit pi-pulse)"""
        param = {}
        param['e'] = self.phase['g']
        param['g'] = self.phase['e']
        self.phase = param

        param = {}
        param['e'] = self.angle['g']
        param['g'] = self.angle['e']
        self.angle = param

        param= {}
        param['e'] = self.alpha['g']
        param['g'] = self.alpha['e']
        self.alpha = param

    def CD_sequence(self, alpha, phi_g, phi_e, tau):
        """
        This is a simple CD implementation that includes amplitude and phse 
        corrections for the return pulses.
        
        """
        phi_diff = phi_g - phi_e
        phi_sum = phi_g + phi_e
        self.displace(alpha)
        self.delay(tau)
        self.displace(-alpha*tf.math.cos(phi_diff/2)*tf.math.exp(1j*phi_sum/2))
        self.flip()
        self.displace(-alpha*tf.math.cos(phi_diff/2)*tf.math.exp(1j*phi_sum/2))
        self.delay(tau)
        self.displace(alpha*tf.math.cos(phi_diff)*tf.math.exp(1j*phi_sum))
        self.rotate('g', -tf.cast(phi_sum, tf.float32))
        self.rotate('e', -tf.cast(phi_sum, tf.float32))



class QuantumCircuit(OscillatorQubit, GKP):
    """
    A one-step circuit for calibration of the CD gate. This requires having a
    reliable calibration of displacements, at least up to the amplitudes used
    in the CD (the amplitude 'beta'). 
    
    """
    def __init__(
        self,
        *args,
        # Required kwargs
        t_gate,
        t_idle,
        # Optional kwargs
        **kwargs):
        """
        Args:
            t_gate (float): Gate time in seconds.
            t_idle (float): Wait time between rounds in seconds.
        """
        self.t_gate = tf.constant(t_gate, dtype=tf.float32)
        self.t_idle = tf.constant(t_idle, dtype=tf.float32)
        self.step_duration = tf.constant(t_gate + t_idle)
    
        super().__init__(*args, **kwargs)
        
        self.analytic_sim = PhaseSpaceSimulator(self.batch_size)

    @property
    def _quantum_circuit_spec(self):
        spec = {'alpha' : specs.TensorSpec(shape=[1], dtype=tf.float32),
                'phi_g' : specs.TensorSpec(shape=[1], dtype=tf.float32),
                'phi_e' : specs.TensorSpec(shape=[1], dtype=tf.float32)}
        return spec

    # @tf.function
    def _quantum_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'alpha' : Tensor([batch_size,1], tf.float32),
                          'phi_g' : Tensor([batch_size,1], tf.float32),
                          'phi_e' : Tensor([batch_size,1], tf.float32))

        Returns: see parent class docs

        """
        alpha = tf.squeeze(tf.cast(action['alpha'], c64))
        phi_g = tf.squeeze(tf.cast(action['phi_g'], c64))
        phi_e = tf.squeeze(tf.cast(action['phi_e'], c64))
        
        beta = sqrt(2*pi) # amplitude of conditional displacement
        
        # randomly prepare a batch of qubits in state g or e
        mask = tf.where(tf.random.uniform([self.batch_size,1])>0.5, 1, -1)
        psi = tf.where(mask==1, psi, tf.linalg.matvec(self.sx, psi))
        
        # return displacement sign depends on qubit state
        alpha_return = 1j * beta / 2 * tf.squeeze(tf.cast(mask, c64))
        
        # simulate sequence of displacements and delays implementing CD gate
        self.analytic_sim.reset()
        self.analytic_sim.CD_sequence(alpha, phi_g, phi_e, self.t_gate/2)
        self.analytic_sim.displace(alpha_return)
        
        alpha = self.analytic_sim.alpha
        angle = self.analytic_sim.angle
        phase = self.analytic_sim.phase
        
        D = self.ctrl(self.displace(alpha['g']), self.displace(alpha['e']))
        R = self.ctrl(self.rotate(-angle['g']), self.rotate(-angle['e']))
        P = self.P[0]*self.phase(phase['g']) + self.P[1]*self.phase(phase['e'])
        
        # This is the final effect on the qubit-cavity system
        psi = tf.linalg.matvec(self.sx, psi)
        psi = tf.linalg.matvec(D, psi)
        psi = tf.linalg.matvec(R, psi)
        psi = tf.linalg.matvec(P, psi)
        
        # flip the qubit conditioned on vacuum in the cavity
        psi = tf.linalg.matvec(self.sx_selective, psi)
        
        # flip those that started in e, so ideally everything will end up in g
        # and a simple sigma_z measurement can be used as reward.
        psi = tf.where(mask==1, psi, tf.linalg.matvec(self.sx, psi))
        
        return psi, psi, tf.ones([self.batch_size,1])

