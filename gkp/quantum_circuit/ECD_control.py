# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:24:51 2021

@author: Vladimir Sivak
"""
import tensorflow as tf
from gkp.tf_env.tf_env import TFEnvironmentQuantumControl
from gkp.tf_env import helper_functions as hf
from tf_agents import specs
from simulator.hilbert_spaces import OscillatorQubit
from simulator.utils import measurement
from math import pi
from tensorflow import complex64 as c64

class QuantumCircuit(OscillatorQubit, TFEnvironmentQuantumControl):
    """
    Universal gate sequence for open-loop unitary control of the oscillator.
    The gate sequence consists of 
        1) qubit rotations in the xy-plane of the Bloch sphere
        2) conditional translations of the oscillator conditioned on qubit
    
    """
    def __init__(self, *args, t_gate, **kwargs):
        """
        Args:
            t_gate (float): Gate time in seconds.
        """
        self.t_gate = tf.constant(t_gate, dtype=tf.float32)
        self.step_duration = self.t_gate
        super().__init__(*args, **kwargs)

    @property
    def _control_circuit_spec(self):
        spec = {'beta'  : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'phi'   : specs.TensorSpec(shape=[2], dtype=tf.float32)}
        return spec

    # remove the decorator to be able to access the class attributes from withing this method
    # @tf.function
    def _control_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'beta'  : Tensor([batch_size,2], tf.float32),
                          'phi'   : Tensor([batch_size,2], tf.float32))

        Returns: see parent class docs

        """
        # Extract parameters
        beta = hf.vec_to_complex(action['beta'])
        phase, angle = action['phi'][:,0], action['phi'][:,1]

        # Construct gates        
        D = self.displace(beta/2.0)
        CD = self.ctrl(D, tf.linalg.adjoint(D))
        R = self.rotate_qb_xy(angle, phase)
        flip = self.rotate_qb_xy(tf.constant(pi), tf.constant(0))
        
        # Apply gates
        psi = tf.linalg.matvec(R, psi)
        psi = tf.linalg.matvec(CD, psi)
        if self._elapsed_steps < self.T-1:
            psi = tf.linalg.matvec(flip, psi)
        
        m = tf.ones((self.batch_size,1))
        # if self._elapsed_steps == self.T-1:
        #     psi, m = measurement(psi, self.P, sample=True)

        return psi, psi, m

