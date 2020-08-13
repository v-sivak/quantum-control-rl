# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:58:30 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from tensorflow.keras.backend import batch_dot

from simulator.utils import normalize


class QuantumTrajectorySim:
    """
    Tensorflow implementation of the Monte Carlo quantum trajectory simulator.
    """

    def __init__(self, Kraus):
        """
        Input:
            Kraus -- dictionary of Kraus operators. Dict keys are integers.
                     K[0] is no-jump operator, the rest are jump operators.
                     Shape of each operator is [b,NH,NH], b is batch size
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

    @tf.function
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
