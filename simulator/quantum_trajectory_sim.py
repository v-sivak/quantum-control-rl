# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:58:30 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
import tensorflow_probability as tfp
from simulator.utils import normalize


class QuantumTrajectorySim:
    """
    TensorFlow implementation of the Monte Carlo quantum trajectory simulator.
    
    """
    def __init__(self, Kraus_operators):
        """
        Args:
            Kraus_operators (dict: Tensor(c64)): dictionary of Kraus operators. 
                By convention, K[0] is no-jump operator, K[i>0] are jump operators.
        """
        self.Kraus_operators = Kraus_operators

    def _step(self, j, psi, steps):
        """ Single Monte Carlo step. """
        traj, p, norm = {}, {}, {}
        cumulant = tf.zeros([psi.shape[0], 1])
        prob = tf.random.uniform([psi.shape[0], 1])
        state = psi
        for i, Kraus in self.Kraus_operators.items():
            # Compute a trajectory for this Kraus operator
            traj[i] = tf.linalg.matvec(Kraus, psi) # shape = [b,N]
            traj[i], p[i] = normalize(traj[i])
            p[i] = tf.math.real(p[i]) # shape = [b,1]
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

        Args:
            psi (Tensor([B1,...Bb,N], c64)): batch of quantum states.
            steps (int): number of steps to run the trajectory

        """
        psi, _ = normalize(psi)
        j = tf.constant(0)
        _, psi_new, steps = tf.while_loop(
            self._cond, self._step, loop_vars=[j, psi, steps]
        )

        # Check for NaN
        mask = tf.math.is_nan(tf.math.real(psi_new))
        psi_new = tf.where(mask, psi, psi_new)
        return psi_new
