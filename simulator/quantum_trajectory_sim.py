# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:58:30 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.backend import batch_dot

from simulator.utils import normalize


class QuantumTrajectorySim:
    """
    Tensorflow implementation of the Monte Carlo quantum trajectory simulator.
    """

    def __init__(self, Kraus):
        """
        Input:
            Kraus -- dictionary of Tensor([N, N], c64) Kraus operators.
                     K[0] is no-jump operator, K[i] with i>0 is jump operator.
        """
        self.Kraus = Kraus

    def _step(self, j, psi, steps):
        """
        One step in the Markov chain.
        """
        traj, p, norm = {}, {}, {}
        cumulant = tf.zeros([psi.shape[0], 1])
        prob = tf.random.uniform([psi.shape[0], 1])
        state = psi
        for i in self.Kraus.keys():
            # Compute a trajectory for this Kraus operator
            traj[i] = tf.linalg.matvec(self.Kraus[i], psi)  # shape = [b,N]
            p[i] = batch_dot(tf.math.conj(traj[i]), traj[i])  # shape = [b,1]
            p[i] = tf.math.real(p[i])
            norm[i] = tf.math.sqrt(p[i]) + 1e-15  # shape = [b,1]
            norm[i] = tf.cast(norm[i], tf.complex64)
            traj[i] = traj[i] / norm[i]
            # Select this trajectory depending on sampled 'prob'
            mask = tf.math.logical_and(prob > cumulant, prob < cumulant + p[i])
            state = tf.where(mask, traj[i], state)
            # Update cumulant
            cumulant += p[i]
        return [j + 1, state, steps]

    def _cond(self, j, psi, steps):
        return tf.less(j, steps)

    def run(self, psi, steps):
        """
        Simulate a batch of trajectories for a number of steps.

        Input:
            psi -- batch of state vectors; shape=[b,NH]
            steps -- number of steps to run the trajectory

        """
        psi = normalize(psi)
        j = tf.constant(0)
        _, psi_new, steps = tf.while_loop(
            self._cond, self._step, loop_vars=[j, psi, steps]
        )

        # Check for NaN
        mask = tf.math.is_nan(tf.math.real(psi_new))
        psi_new = tf.where(mask, psi, psi_new)
        return psi_new

    def measure(self, psi, Kraus, sample):
        """
        Batch measurement projection.

        Input:
            psi -- batch of states; shape=[batch_size, NH]
            Kraus -- dictionary of Kraus operators corresponding to 2 different
                     qubit measurement outcomes. Shape of each operator is
                     [b,NH,NH], where b is batch size
            sample -- bool flag to sample or return expectation value

        Output:
            psi -- batch of collapsed states; shape=[batch_size,NH]
            obs -- measurement outcomes; shape=[batch_size,1]

        """
        collapsed, p = {}, {}
        for i in Kraus.keys():
            collapsed[i] = tf.linalg.matvec(Kraus[i], psi)
            p[i] = batch_dot(tf.math.conj(collapsed[i]), collapsed[i])
            p[i] = tf.math.real(p[i])

        if sample:
            obs = tfp.distributions.Bernoulli(probs=p[1] / (p[0] + p[1])).sample()
            psi = tf.where(obs == 1, collapsed[1], collapsed[0])
            obs = 1 - 2 * obs  # convert to {-1,1}
            obs = tf.cast(obs, dtype=tf.float32)
            return psi, obs
        else:
            return psi, p[0] - p[1]
