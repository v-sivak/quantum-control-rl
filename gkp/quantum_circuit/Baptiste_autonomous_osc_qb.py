# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:52:25 2020

@author: Vladimir Sivak

Qubit is included in the Hilbert space. Simulation is done with a gate-based 
approach to quantum circuits.
"""

import tensorflow as tf
from tensorflow import complex64 as c64
from gkp.gkp_tf_env.gkp_tf_env import GKP
from gkp.gkp_tf_env import helper_functions as hf
from tf_agents import specs
from simulator.hilbert_spaces import OscillatorQubit
from simulator.utils import measurement


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
        phi = tf.squeeze(action['phi'])

        # Construct gates
        Phase = self.rotate_qb_z(phi)
        T, CT = {}, {}
        T['b'] = self.translate(beta/4.0) # 4 because it will be troterized
        T['e'] = self.translate(epsilon/2.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])
        CT['e'] = self.ctrl(tf.linalg.adjoint(T['e']), T['e'])

        # Feedback translation
        psi_cached = psi
        # Between-round wait time
        psi = self.simulate(psi_cached, self.t_idle)
        # Rotate qubit to |+> state
        psi = tf.linalg.matvec(self.hadamard, psi)
        # Troterized conditional translation
        psi = tf.linalg.matvec(CT['b'], psi)
        psi = self.simulate(psi, self.t_gate)
        psi = tf.linalg.matvec(CT['b'], psi)
        # Qubit rotation
        psi = tf.linalg.matvec(self.rxp, psi)
        # Conditional translation
        psi = tf.linalg.matvec(CT['e'], psi)
        # Qubit rotation
        psi = tf.linalg.matvec(self.rxm, psi)
        # Troterized conditional translation
        psi = tf.linalg.matvec(CT['b'], psi)
        psi = self.simulate(psi, self.t_gate)
        psi = tf.linalg.matvec(CT['b'], psi)
        # Qubit gates
        psi = tf.linalg.matvec(Phase, psi)
        psi = tf.linalg.matvec(self.hadamard, psi)
        # Readout of finite duration
        psi = self.simulate(psi, self.t_read)
        psi, msmt = measurement(psi, self.P)
        psi = self.simulate(psi, self.t_read)
        # Feedback delay
        psi = self.simulate(psi, self.t_feedback)
        # Flip qubit conditioned on the measurement
        psi_final = tf.where(msmt==1, psi, tf.linalg.matvec(self.sx, psi))

        return psi_final, psi_cached, msmt
