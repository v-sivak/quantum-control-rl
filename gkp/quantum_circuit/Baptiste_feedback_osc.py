# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:18:11 2020

@author: Vladimir Sivak

Abstract away the qubit and use Kraus maps formalism to efficiently simulate 
operations on the oscillator Hilbert space.
"""

import tensorflow as tf
from numpy import sqrt
from tensorflow.keras.backend import batch_dot
from gkp.tf_env.tf_env import GKP
from gkp.tf_env import helper_functions as hf
from tf_agents import specs
from simulator.hilbert_spaces import Oscillator
from simulator.utils import measurement


class QuantumCircuit(Oscillator, GKP):
    """    
    This is a protocol proposed by Baptiste https://arxiv.org/abs/2009.07941
    It essentially combines trimming and sharpening in a single round.
    
    """
    def __init__(
        self,
        *args,
        # Required kwargs
        t_gate,
        t_read,
        t_feedback,
        t_idle,
        # Optional kwargs
        **kwargs):
        """
        Args:
            t_gate (float): Gate time in seconds.
            t_read (float): Readout time in seconds.
            t_feedback (float): Feedback delay in seconds.
            t_idle (float): Wait time between rounds in seconds.
        """
        self.t_round = tf.constant(t_gate + t_read, dtype=tf.float32)
        self.t_feedback = tf.constant(t_feedback, dtype=tf.float32)
        self.t_idle = tf.constant(t_idle, dtype=tf.float32)
        self.step_duration = tf.constant(t_gate + t_read + t_feedback + t_idle)
        super().__init__(*args, **kwargs)

    @property
    def _quantum_circuit_spec(self):
        spec = {'alpha'   : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'beta'    : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'epsilon' : specs.TensorSpec(shape=[2], dtype=tf.float32)}
        return spec

    @tf.function
    def _quantum_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'alpha'   : Tensor([batch_size,2], tf.float32),
                          'beta'    : Tensor([batch_size,2], tf.float32),
                          'epsilon' : Tensor([batch_size,2], tf.float32))

        Returns: see parent class docs

        """
        # extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        epsilon = hf.vec_to_complex(action['epsilon'])

        Kraus = {}
        T = {}
        T['a'] = self.translate(alpha)
        T['+b'] = self.translate(beta/2.0)
        T['-b'] = tf.linalg.adjoint(T['+b'])
        T['+e'] = self.translate(epsilon/2.0)
        T['-e'] = tf.linalg.adjoint(T['+e'])


        chunk1 = batch_dot(T['-e'], T['-b']) - 1j*batch_dot(T['-e'], T['+b'])
        chunk2 = batch_dot(T['+e'], T['-b']) + 1j*batch_dot(T['+e'], T['+b'])

        Kraus[0] = 1/2/sqrt(2)*(chunk1 + chunk2)
        Kraus[1] = 1/2/sqrt(2)*(chunk1 - chunk2)

        psi = self.simulate(psi, self.t_feedback)
        psi_cached = batch_dot(T['a'], psi)
        psi = self.simulate(psi_cached, self.t_round + self.t_idle)
        psi_final, msmt = measurement(psi, Kraus)

        return psi_final, psi_cached, msmt
