# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:52:25 2020

@author: Vladimir Sivak

Qubit is included in the Hilbert space. Simulation is done with a gate-based 
approach to quantum circuits.
"""

import tensorflow as tf
from tensorflow import complex64 as c64
from tensorflow.keras.backend import batch_dot
from gkp.gkp_tf_env.gkp_tf_env import GKP
from gkp.gkp_tf_env import helper_functions as hf
from tf_agents import specs
from simulator.hilbert_spaces import OscillatorQubit
from simulator.utils_v2 import measurement


class QuantumCircuit(OscillatorQubit, GKP):
    """
    This is a protocol proposed by Baptiste https://arxiv.org/abs/2009.07941
    It essentially combines trimming and sharpening in a single round.
    Autonomous version (no feedback required, but measurements still
    needed to evacuate entropy).
    
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
        self.t_read = tf.constant(t_read / 2, dtype=tf.float32)  # Split read time before/after meas.
        self.t_gate = tf.constant(t_gate, dtype=tf.float32)
        self.t_feedback = tf.constant(t_feedback, dtype=tf.float32)
        self.t_idle = tf.constant(t_idle, dtype=tf.float32)
        self.step_duration = tf.constant(t_gate + t_read + t_feedback + t_idle)
        super().__init__(*args, **kwargs)

    @property
    def _quantum_circuit_spec(self):
        spec = {'alpha'   : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'beta'    : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'epsilon' : specs.TensorSpec(shape=[2], dtype=tf.float32),
                'phi'     : specs.TensorSpec(shape=[1], dtype=tf.float32)}
        return spec

    @tf.function
    def _quantum_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'alpha'   : Tensor([batch_size,2], tf.float32),
                          'beta'    : Tensor([batch_size,2], tf.float32),
                          'epsilon' : Tensor([batch_size,2], tf.float32,
                          'phi'     : Tensor([batch_size,1])))

        Returns: see parent class docs

        """
        # extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        epsilon = hf.vec_to_complex(action['epsilon'])
        phi = action['phi']

        # Construct gates
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        sx = tf.stack([self.sx]*self.batch_size)
        I = tf.stack([self.I]*self.batch_size)
        Phase = self.rotate_qb_z(tf.squeeze(phi))
        Rxp = tf.stack([self.rxp]*self.batch_size)
        Rxm = tf.stack([self.rxm]*self.batch_size)
        T, CT = {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/4.0)
        T['e'] = self.translate(epsilon/2.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])
        CT['e'] = self.ctrl(tf.linalg.adjoint(T['e']), T['e'])

        # Feedback translation
        psi_cached = batch_dot(T['a'], psi)
        # Between-round wait time
        psi = self.simulate(psi_cached, self.t_idle)
        # Qubit gates
        psi = batch_dot(Hadamard, psi)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.simulate(psi, self.t_gate)
        psi = batch_dot(CT['b'], psi)
        # Qubit rotation
        psi = batch_dot(Rxp, psi)
        # Conditional translation
        psi = batch_dot(CT['e'], psi)
        # Qubit rotation
        psi = batch_dot(Rxm, psi)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.simulate(psi, self.t_gate)
        psi = batch_dot(CT['b'], psi)
        # Qubit gates
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        # Readout of finite duration
        psi = self.simulate(psi, self.t_read)
        psi, msmt = measurement(psi, self.P)
        psi = self.simulate(psi, self.t_read)
        # Feedback delay
        psi = self.simulate(psi, self.t_feedback)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((msmt==1), c64)
        psi_final += batch_dot(sx, psi) * tf.cast((msmt==-1), c64)

        return psi_final, psi_cached, msmt
