# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:19:56 2020

@author: Vladimir Sivak

Qubit is included in the Hilbert space. Simulation is done with a gate-based 
approach to quantum circuits.
"""
import tensorflow as tf
from numpy import pi
from rl_tools.tf_env.tf_env import GKP
from rl_tools.tf_env import helper_functions as hf
from tf_agents import specs
from simulator.hilbert_spaces import OscillatorQubit
from simulator.utils import measurement, normalize

class QuantumCircuit(OscillatorQubit, GKP):
    """
    Universal gate sequence for open-loop unitary control of the oscillator
    in the large-chi regime. 
    
    The gate sequence consists of 
        1) oscillator displacement
        2) selective number-dependent arbitrary phase (SNAP) gate
        3) oscillator return displacement
    
    """
    def __init__(
        self,
        *args,
        # Required kwargs
        t_gate,
        # Optional kwargs
        **kwargs):
        """
        Args:
            t_gate (float): Gate time in seconds.
        """
        self.t_gate = tf.constant(t_gate, dtype=tf.float32)
        self.step_duration = self.t_gate
        self.bit_string = None
        self.norms = []
        super().__init__(*args, **kwargs)

    @property
    def _control_circuit_spec(self):
        spec = {'alpha' : specs.TensorSpec(shape=[2], dtype=tf.float32),
                'theta' : specs.TensorSpec(shape=[7], dtype=tf.float32)}
        return spec

    # remove the decorator to be able to access the class attributes from withing this method
    # @tf.function
    def _control_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'alpha' : Tensor([batch_size,2], tf.float32),
                          'theta' : Tensor([batch_size,10], tf.float32))

        Returns: see parent class docs

        """
        # Extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        theta = action['theta']

        # Build gates
        displace = self.displace(alpha)
        snap = self.SNAP_miscalibrated(theta)

        # Apply gates
        psi = tf.linalg.matvec(displace, psi)
        psi = tf.linalg.matvec(snap, psi)
        psi = tf.linalg.matvec(tf.linalg.adjoint(displace), psi)

        # Either implement a measurement with a random qubit projection, or 
        # project on the specified 'self.bit_string' sequence of qubit states.
        if self.bit_string is not None:
            s = int(self.bit_string[self._elapsed_steps])
            psi = tf.linalg.matvec(self.P[s], psi)
            if s == 1: psi = tf.linalg.matvec(self.sx, psi)
            psi, norm = normalize(psi)
            self.norms.append(norm)
            msmt = tf.ones((self.batch_size,1)) * (1 if s==0 else -1)
        else: # Readout with feedback to flip the qubit
            psi, msmt = measurement(psi, self.P)
            psi = tf.where(msmt==1, psi, tf.linalg.matvec(self.sx, psi))

        return psi, psi, msmt

