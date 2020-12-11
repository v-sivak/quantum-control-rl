"""
Abstract base class to define a specific simulator Hilbert space

Created on Sun Jul 26 19:39:18 2020

@author: Henry Liu
"""
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.backend import batch_dot
from simulator.quantum_trajectory_sim import QuantumTrajectorySim
from simulator.diffusion_channel_sim import DiffusionChannelSim


class HilbertSpace(ABC):
    """
    Abstract base class which intializes a Monte Carlo simulator for a particular
    Hilbert space. The space is defined by the subclass, which implements a set of
    operators on the space, a Hamiltonian, and a set of jump operators.
    """

    def __init__(self, *args, channel, N, discrete_step_duration, diffusion_rate, **kwargs):
        """
        Args:
            channel (str): either 'diffusion' or 'quantum_jumps' error channel
            N (int): Size of the oscillator Hilbert space
            discrete_step_duration (float): Simulator time discretization in seconds.
            diffusion_rate (float): Rate of diffusion in 1/s.
        """
        # Tensor ops acting on oscillator Hilbert space
        self._define_fixed_operators(N)
        
        # Initialize quantum trajectories simulator
        if channel == 'quantum_jumps':
            self.mcsim = QuantumTrajectorySim(self._kraus_ops(discrete_step_duration))
            def simulate(psi, time):
                # TODO: fix the rounding issue
                steps = tf.cast(time / discrete_step_duration, dtype=tf.int32)
                return self.mcsim.run(psi, steps)
        
        if channel == 'diffusion':
            self.mcsim = DiffusionChannelSim(self.translate)
            def simulate(psi, time):
                diffusion_sigma = tf.math.sqrt(diffusion_rate * time)
                return self.mcsim.run(psi, diffusion_sigma)

        self.simulate = tf.function(simulate)
        super().__init__(*args, **kwargs)
    

    @abstractmethod
    def _define_fixed_operators(self, N):
        """
        Fixed operators on this Hilbert space, to be defined by the subclass.

        For a (trivial) example: the identity operator. "Batch" operators, which are
        generated from an input parameter (e.g. displacement) are currently defined
        in a separate mixin.

        Args:
            N (int): Size of the **oscillator** Hilbert space truncation.
        """
        pass

    @property
    @abstractmethod
    def _hamiltonian(self):
        """
        System Hamiltonian, to be defined by the subclass.
        """
        pass

    @property
    @abstractmethod
    def _collapse_operators(self):
        """
        Kraus jump operators, to be defined by the subclass.
        """
        pass

    def _kraus_ops(self, dt):
        """
        Create kraus ops for free evolution simulator

        Args:
            dt (float): Discretized time step of simulator
        """
        Kraus = {}
        Kraus[0] = self.I - 1j * self._hamiltonian * dt
        for i, c in enumerate(self._collapse_operators):
            Kraus[i + 1] = tf.cast(tf.sqrt(dt), dtype=tf.complex64) * c
            Kraus[0] -= 1 / 2 * tf.linalg.matmul(c, c, adjoint_a=True) * dt

        return Kraus
