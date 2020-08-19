"""
Abstract base class to define a specific simulator Hilbert space

Created on Sun Jul 26 19:39:18 2020

@author: Henry Liu
"""
from abc import ABC, abstractmethod

import tensorflow as tf

from simulator.quantum_trajectory_sim import QuantumTrajectorySim


class SimulatorHilbertSpace(ABC):
    """
    Abstract base class which intializes a Monte Carlo simulator for a particular
    Hilbert space. The space is defined by the subclass, which implements a set of
    operators on the space, a Hamiltonian, and a set of jump operators.
    """

    def __init__(self, *args, N, discrete_step_duration, **kwargs):
        """
        Args:
            N (int): Size of the oscillator Hilbert space
            discrete_step_duration (float): Simulator time discretization in seconds.
        """
        # Tensor ops acting on oscillator Hilbert space
        self._define_fixed_operators(N)

        # Initialize quantum trajectories simulator
        self.mcsim = QuantumTrajectorySim(self._kraus_ops(discrete_step_duration))
        self._dt = tf.constant(discrete_step_duration)

        super().__init__(*args, **kwargs)

    @tf.function
    def simulate(self, psi, time):
        """
        Evolves the wavefunction `psi` for `time` using the Monte Carlo simulator with
        previously defined Kraus operators. This is mostly a helper function to
        abstract the time discretization of the simulator from the caller. Note,
        however, that discretization *does* impart a rounding effect.

        Args:
            psi (Tensor([batch_size, N], c64)): The (batched) WF to evolve.
            time (float): Time in seconds to run the simulation forward by.

        Returns:
            Tensor([batch_size, N], c64: The evolved wavefunction after time/dt steps.
        """
        steps = tf.cast(time / self._dt, dtype=tf.int32)  # TODO: fix the rounding issue
        return self.mcsim.run(psi, steps)

    @tf.function
    def measure(self, psi, Kraus, sample=True):
        """
        Batch measurement projection using the Monte Carlo simulator.

        Args:
            psi (Tensor([batch_size, N], c64)):
                The (batched) WF to measure projectively.
            Kraus ({0: Tensor([batch_size, N, N], c64), 1: Tensor(...)}):
                Dictionary of Kraus operators corresponding to 2 different qubit
                measurement outcomes.
            sample (bool, optional):
                Sample or return expectation value. Defaults to True.

        Returns:
            (Tensor([batch_size, N], c64), Tensor([batch_size, 1], c64)):
                0 -- batch of collapsed states
                1 -- measurement outcomes
        """
        return self.mcsim.measure(psi, Kraus, sample)

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
