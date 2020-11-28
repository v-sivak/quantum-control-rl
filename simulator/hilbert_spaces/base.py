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
from simulator.utils import normalize

class HilbertSpace(ABC):
    """
    Abstract base class which intializes a Monte Carlo simulator for a particular
    Hilbert space. The space is defined by the subclass, which implements a set of
    operators on the space, a Hamiltonian, and a set of jump operators.
    """

    def __init__(self, *args, discrete_step_duration, diffusion_rate, **kwargs):
        """
        Args:
            discrete_step_duration (float): Simulator time discretization in seconds.
            diffusion_rate (float): Rate of diffusion in 1/s.
        """
        # Tensor ops acting on oscillator Hilbert space
        self._define_operators()
        
        # Initialize quantum trajectories simulator
        self.mcsim = QuantumTrajectorySim(self._kraus_ops(discrete_step_duration))
        def simulate(psi, time):
            # TODO: fix the rounding issue
            steps = tf.cast(time / discrete_step_duration, dtype=tf.int32)
            return self.mcsim.run(psi, steps)
        self.simulate = tf.function(simulate)

        super().__init__(*args, **kwargs)
    

    @abstractmethod
    def _define_operators(self):
        """
        Create operators on this Hilbert space. To be defined by the subclass.

        Example:
            self.I = operators.identity(self.N)
            self.p = operators.momentum(self.N)
            self.displace = operators.DisplacementOperator(self.N)
        """
        pass

    @property
    @abstractmethod
    def _hamiltonian(self):
        """
        System Hamiltonian (LinearOperator), to be defined by the subclass.
        """
        pass

    @property
    @abstractmethod
    def _dissipator(self):
        """
        List of collapse operators (LinearOperator), to be defined by the subclass.
        """
        pass

    def _kraus_ops(self, dt):
        """
        Create kraus ops for free evolution simulator

        Args:
            dt (float): Discretized time step of simulator
        """
        Kraus = {}
        Kraus[0] = self.I.to_dense() - 1j * self._hamiltonian * dt
        for i, c in enumerate(self._dissipator):
            Kraus[i + 1] = tf.cast(tf.sqrt(dt), dtype=tf.complex64) * c
            Kraus[0] -= 1 / 2 * tf.linalg.matmul(c, c, adjoint_a=True) * dt

        return Kraus
