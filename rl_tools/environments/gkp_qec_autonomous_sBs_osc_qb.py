# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:52:25 2020

@author: Vladimir Sivak

Qubit is included in the Hilbert space. Simulation is done with a gate-based 
approach to quantum circuits.
"""

import tensorflow as tf
from tensorflow import complex64 as c64
from rl_tools.tf_env.tf_env import TFEnvironmentQuantumControl
from rl_tools.tf_env import helper_functions as hf
from tf_agents import specs
from simulator.hilbert_spaces import OscillatorQubit
from simulator.utils import measurement


class QuantumCircuit(OscillatorQubit, TFEnvironmentQuantumControl):
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
    def _control_circuit_spec(self):
        spec = {'beta' : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'eps1' : specs.TensorSpec(shape=[2], dtype=tf.float32),
                'eps2' : specs.TensorSpec(shape=[2], dtype=tf.float32),
                'phi'  : specs.TensorSpec(shape=[1], dtype=tf.float32),
                'theta' : specs.TensorSpec(shape=[1], dtype=tf.float32)}
        return spec

    @tf.function
    def _control_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'beta'    : Tensor([batch_size,2], tf.float32),
                          'eps1' : Tensor([batch_size,2], tf.float32),
                          'eps2' : Tensor([batch_size,2], tf.float32),
                          'phi'     : Tensor([batch_size,1]),
                          'theta'   : Tensor([batch_size,1]))

        Returns: see parent class docs

        """
        # extract parameters
        beta = hf.vec_to_complex(action['beta'])
        eps1 = hf.vec_to_complex(action['eps1'])
        eps2 = hf.vec_to_complex(action['eps2'])
        phi = tf.squeeze(action['phi'])

        # Construct gates
        Phase = self.rotate_qb_z(phi)
        Rotation = self.rotate(action['theta'])
        T, CT = {}, {}
        T['b'] = self.translate(beta/4.0) # 4 because it will be troterized
        T['e1'] = self.translate(eps1/2.0)
        T['e2'] = self.translate(eps2/2.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])
        CT['e1'] = self.ctrl(tf.linalg.adjoint(T['e1']), T['e1'])
        CT['e2'] = self.ctrl(tf.linalg.adjoint(T['e2']), T['e2'])

        psi_cached = psi
        # Between-round wait time
        psi = self.simulate(psi, self.t_idle)
        psi = tf.linalg.matvec(Rotation, psi)
        # Rotate qubit to |+> state
        psi = tf.linalg.matvec(self.hadamard, psi)
        # Conditional translation
        psi = tf.linalg.matvec(CT['e1'], psi)
        # Qubit rotation
        psi = tf.linalg.matvec(self.rxp, psi)
        # Troterized conditional translation
        psi = tf.linalg.matvec(CT['b'], psi)
        psi = self.simulate(psi, self.t_gate)
        psi = tf.linalg.matvec(CT['b'], psi)
        # Qubit rotation
        psi = tf.linalg.matvec(self.rxm, psi)
        # Conditional translation
        psi = tf.linalg.matvec(CT['e2'], psi)
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

