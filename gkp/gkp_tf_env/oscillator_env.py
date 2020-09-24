# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:30:01 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
from numpy import sqrt
from tensorflow.keras.backend import batch_dot

from gkp.gkp_tf_env.gkp_tf_env import GKP
from gkp.gkp_tf_env import helper_functions as hf

from simulator.hilbert_spaces import Oscillator
from simulator.utils import normalize


class OscillatorGKP(Oscillator, GKP):
    """
    This class inherits simulation-independent functionality from the GKP
    class and implements simulation by abstracting away the qubit and using
    Kraus maps formalism to rather efficiently simulate operations on the
    oscillator Hilbert space.
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
        N=100,
        channel='quantum_jumps', #TODO: why do we need N and channel here? 
        **kwargs
    ):
        """
        Args:
            t_gate (float): Gate time in seconds.
            t_read (float): Readout time in seconds.
            t_feedback (float): Feedback delay in seconds.
            t_idle (float): Wait time between rounds in seconds.
            N (int, optional): Size of oscillator Hilbert space. Defaults to 100.
            channel (str, optional): model of the error channel, either 'diffusion'
                    or 'quantum_jumps'.
        """
        self._N = N
        self.t_round = tf.constant(t_gate + t_read, dtype=tf.float32)
        self.t_feedback = tf.constant(t_feedback, dtype=tf.float32)
        self.t_idle = tf.constant(t_idle, dtype=tf.float32)
        self.step_duration = tf.constant(t_gate + t_read + t_feedback + t_idle)
        super().__init__(*args, N=N, channel=channel, **kwargs)

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return False

    @tf.function
    def quantum_circuit_v1(self, psi, action):
        """
        Apply Kraus map version 1. In this version conditional translation by
        'beta' is not symmetric (translates if qubit is in '1')

        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'phi'

        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)

        """
        # extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        phi = action['phi']

        Kraus = {}
        T = {'a' : self.translate(alpha),
             'b' : self.translate(beta)}
        I = tf.stack([self.I]*self.batch_size)
        Kraus[0] = 1/2*(I + self.phase(phi)*T['b'])
        Kraus[1] = 1/2*(I - self.phase(phi)*T['b'])

        psi = self.simulate(psi, self.t_feedback)
        psi_cached = batch_dot(T['a'], psi)
        psi = self.simulate(psi_cached, self.t_round + self.t_idle)
        psi = normalize(psi)
        psi_final, obs = self.measure(psi, Kraus)

        return psi_final, psi_cached, obs

    @tf.function
    def quantum_circuit_v2(self, psi, action):
        """
        Apply Kraus map version 2. In this version conditional translation by
        'beta' is symmetric (translates by +-beta/2 controlled by the qubit)

        Input:
            action -- batch of actions; shape=[batch_size,5]

        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)

        """
        # extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        phi = action['phi']
        Rotation = self.rotate(action['theta'])

        Kraus = {}
        T = {'a' : self.translate(alpha),
             'b' : self.translate(beta/2.0)}
        Kraus[0] = 1/2*(tf.linalg.adjoint(T['b']) + self.phase(phi)*T['b'])
        Kraus[1] = 1/2*(tf.linalg.adjoint(T['b']) - self.phase(phi)*T['b'])

        psi = self.simulate(psi, self.t_feedback)
        psi = batch_dot(T['a'], psi)
        psi_cached = batch_dot(Rotation, psi)
        psi = self.simulate(psi_cached, self.t_round + self.t_idle)
        psi = normalize(psi)
        psi_final, obs = self.measure(psi, Kraus)

        return psi_final, psi_cached, obs


    @tf.function
    def quantum_circuit_v3(self, psi, action):
        """
        Apply Kraus map version 3. This is a protocol proposed by Baptiste.
        It essentially combines trimming and sharpening in a single round.
        Trimming is controlled by 'epsilon'.

        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'epsilon', 'phi'

        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=[batch_size,1]

        """
        # extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        epsilon = hf.vec_to_complex(action['epsilon'])
        phi = action['phi']

        Kraus = {}
        T = {}
        T['a'] = self.translate(alpha)
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

        psi = self.simulate(psi, self.t_feedback)
        psi_cached = batch_dot(T['a'], psi)
        psi = self.simulate(psi_cached, self.t_round + self.t_idle)
        psi = normalize(psi)
        psi_final, obs = self.measure(psi, Kraus)

        return psi_final, psi_cached, obs


    @tf.function
    def quantum_circuit_v4(self, psi, action):
        """
        Apply Kraus map version 4. This is a protocol proposed by Baptiste.
        It essentially combines trimming and sharpening in a single round.
        Trimming is controlled by 'epsilon'. This is similar to 'v3', but
        the last conditional displacement gate is replaced with classicaly
        conditioned feedback.

        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'epsilon'

        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=[batch_size,1]

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
        psi = normalize(psi)
        psi_final, obs = self.measure(psi, Kraus)

        return psi_final, psi_cached, obs


    @tf.function
    def phase_estimation(self, psi, beta, angle, sample=False):
        """
        One round of phase estimation.

        Input:
            psi -- batch of state vectors; shape=[batch_size,N]
            beta -- translation amplitude. shape=(batch_size,)
            angle -- angle along which to measure qubit. shape=(batch_size,)
            sample -- bool flag to sample or return expectation value

        Output:
            psi -- batch of collapsed states if sample==True, otherwise same
                   as input psi; shape=[batch_size,N]
            z -- batch of measurement outcomes if sample==True, otherwise
                 batch of expectation values of qubit sigma_z.

        """
        Kraus = {}
        I = tf.stack([self.I]*self.batch_size)
        T_b = self.translate(beta)
        Kraus[0] = 1/2*(I + self.phase(angle)*T_b)
        Kraus[1] = 1/2*(I - self.phase(angle)*T_b)

        psi = normalize(psi)
        return self.measure(psi, Kraus, sample)
