# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:41:16 2020

@author: Vladimir Sivak

Abstract away the qubit and use Kraus maps formalism to efficiently simulate 
operations on the oscillator Hilbert space.
"""

import tensorflow as tf
from numpy import sqrt
from tensorflow.keras.backend import batch_dot
from gkp.gkp_tf_env.gkp_tf_env import GKP
from gkp.gkp_tf_env import helper_functions as hf
from tf_agents import specs
from simulator.hilbert_spaces import Oscillator
from simulator.utils_v2 import measurement


class QuantumCircuit(Oscillator, GKP):
    """    
    This is a protocol proposed by Baptiste https://arxiv.org/abs/2009.07941
    It essentially combines trimming and sharpening in a single round.
    This is an autonomous version (no feedback required, but measurements are
    needed to evacuate entropy) of the Big-small-Big protocol.
    
    """
    def __init__(
        self,
        *args,
        # Required kwargs
        t_gate,
        t_read,
        t_idle,
        # Optional kwargs
        **kwargs):
        """
        Args:
            t_gate (float): Gate time in seconds.
            t_read (float): Readout time in seconds.
            t_idle (float): Wait time between rounds in seconds.
        """
        self.t_round = tf.constant(t_gate + t_read, dtype=tf.float32)
        self.t_idle = tf.constant(t_idle, dtype=tf.float32)
        self.step_duration = tf.constant(t_gate + t_read + t_idle)
        super().__init__(*args, **kwargs)

    @property
    def _quantum_circuit_spec(self):
        spec = {'beta'    : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'epsilon' : specs.TensorSpec(shape=[2], dtype=tf.float32),
                'phi'     : specs.TensorSpec(shape=[1], dtype=tf.float32)}
        return spec

    @tf.function
    def _quantum_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'beta'    : Tensor([batch_size,2], tf.float32),
                          'epsilon' : Tensor([batch_size,2], tf.float32,
                          'phi'     : Tensor([batch_size,1])))

        Returns: see parent class docs

        """
        # extract parameters
        beta = hf.vec_to_complex(action['beta'])
        epsilon = hf.vec_to_complex(action['epsilon'])
        phi = action['phi']

        Kraus = {}
        T = {}
        T['+b'] = self.translate(beta/2.0)
        T['-b'] = tf.linalg.adjoint(T['+b'])
        T['+e'] = self.translate(epsilon/2.0)
        T['-e'] = tf.linalg.adjoint(T['+e'])


        chunk1 = 1j*batch_dot(T['-b'], batch_dot(T['+e'], T['+b'])) \
                - 1j*batch_dot(T['-b'], batch_dot(T['-e'], T['+b'])) \
                + batch_dot(T['-b'], batch_dot(T['-e'], T['-b'])) \
                + batch_dot(T['-b'], batch_dot(T['+e'], T['-b']))

        chunk2 = 1j*batch_dot(T['+b'], batch_dot(T['-e'], T['-b'])) \
                - 1j*batch_dot(T['+b'], batch_dot(T['+e'], T['-b'])) \
                + batch_dot(T['+b'], batch_dot(T['-e'], T['+b'])) \
                + batch_dot(T['+b'], batch_dot(T['+e'], T['+b']))

        Kraus[0] = 1/4*(chunk1 + self.phase(phi)*chunk2)
        Kraus[1] = 1/4*(chunk1 - self.phase(phi)*chunk2)

        psi = self.simulate(psi, self.t_round + self.t_idle)
        psi_final, msmt = measurement(normalize(psi), Kraus)

        return psi_final, psi_final, msmt
