# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:30:36 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
from tensorflow import complex64 as c64
from tensorflow.keras.backend import batch_dot

from gkp.gkp_tf_env.gkp_tf_env import GKP
from gkp.gkp_tf_env import helper_functions as hf
from simulator.hilbert_spaces import OscillatorQubit
from simulator.utils import normalize


class OscillatorQubitGKP(OscillatorQubit, GKP):
    """
    This class inherits simulation-independent functionality from the GKP
    class and implements simulation by including the qubit in the Hilbert
    space and using gate-based approach to quantum circuits.
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
        **kwargs
    ):
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


    @tf.function
    def quantum_circuit_v1(self, psi, action):
        """
        Apply sequenct of quantum gates version 1. In this version conditional
        translation by 'beta' is not symmetric (translates if qubit is in '1')

        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'phi'

        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)

        """
        # Extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        phi = action['phi']

        # Construct gates
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        sx = tf.stack([self.sx]*self.batch_size)
        I = tf.stack([self.I]*self.batch_size)
        Phase = self.rotate_qb(phi, axis='z')
        T, CT = {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/2.0)
        CT['b'] = self.ctrl(I, T['b'])

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
        # Qubit gates
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        # Readout of finite duration
        psi = self.simulate(psi, self.t_read)
        psi, obs = self.measure(psi, self.P)
        psi = self.simulate(psi, self.t_read)
        # Feedback delay
        psi = self.simulate(psi, self.t_feedback)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((obs==1), c64)
        psi_final += batch_dot(sx, psi) * tf.cast((obs==-1), c64)

        return psi_final, psi_cached, obs

    @tf.function
    def quantum_circuit_v2(self, psi, action):
        """
        Apply sequenct of quantum gates version 2. In this version conditional
        translation by 'beta' is symmetric (translates by +-beta/2 controlled
        by the qubit)

        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'phi'

        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)

        """
        # Extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        phi = action['phi']

        # Construct gates
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        sx = tf.stack([self.sx]*self.batch_size)
        I = tf.stack([self.I]*self.batch_size)
        Phase = self.rotate_qb(phi, axis='z')
        Rotation = self.rotate(action['theta'])
        T, CT = {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/4.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])

        # Feedback translation
        psi = batch_dot(T['a'], psi)
        psi_cached = batch_dot(Rotation, psi)
        # Between-round wait time
        psi = self.simulate(psi_cached, self.t_idle)
        # Qubit gates
        psi = batch_dot(Hadamard, psi)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.simulate(psi, self.t_gate)
        psi = batch_dot(CT['b'], psi)
        # Qubit gates
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        # Readout of finite duration
        psi = self.simulate(psi, self.t_read)
        psi, obs = self.measure(psi, self.P)
        psi = self.simulate(psi, self.t_read)
        # Feedback delay
        psi = self.simulate(psi, self.t_feedback)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((obs==1), c64)
        psi_final += batch_dot(sx, psi) * tf.cast((obs==-1), c64)

        return psi_final, psi_cached, obs


    @tf.function
    def quantum_circuit_v3(self, psi, action):
        """
        Apply sequenct of quantum gates version 3. This is a protocol proposed
        by Baptiste. It essentially combines trimming and sharpening in a
        single round. Trimming is controlled by 'epsilon'.

        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'epsilon', 'phi'

        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)

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
        Phase = self.rotate_qb(phi, axis='z')
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
        psi, obs = self.measure(psi, self.P)
        psi = self.simulate(psi, self.t_read)
        # Feedback delay
        psi = self.simulate(psi, self.t_feedback)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((obs==1), c64)
        psi_final += batch_dot(sx, psi) * tf.cast((obs==-1), c64)

        return psi_final, psi_cached, obs


    @tf.function
    def quantum_circuit_v4(self, psi, action):
        """
        Apply sequence of quantum gates version 4. This is a protocol proposed
        by Baptiste. It essentially combines trimming and sharpening in a
        single round. Trimming is controlled by 'epsilon'. This is similar to
        'v3', but the last conditional displacement gate is replaced with
        classicaly conditioned feedback.

        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'epsilon'

        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)

        """
        # extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        epsilon = hf.vec_to_complex(action['epsilon'])

        # Construct gates
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        sx = tf.stack([self.sx]*self.batch_size)
        I = tf.stack([self.I]*self.batch_size)
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
        # Readout of finite duration
        psi = self.simulate(psi, self.t_read)
        psi, obs = self.measure(psi, self.P)
        psi = self.simulate(psi, self.t_read)
        # Feedback delay
        psi = self.simulate(psi, self.t_feedback)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((obs==1), c64)
        psi_final += batch_dot(sx, psi) * tf.cast((obs==-1), c64)

        return psi_final, psi_cached, obs
