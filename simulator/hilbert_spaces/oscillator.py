"""
Simulator for an oscillator (N) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
from numpy import pi
import tensorflow as tf

from simulator.operators import identity, destroy, create, position, momentum, num
from .base import SimulatorHilbertSpace


class Oscillator(SimulatorHilbertSpace):
    """
    Define all relevant operators as tensorflow tensors of shape [N,N].
    Methods need to take care of batch dimension explicitly.

    Initialize tensorflow quantum trajectory simulator. This is used to
    simulate decoherence, dephasing, Kerr etc using quantum jumps.
    """

    def __init__(self, *args, K_osc, T1_osc, **kwargs):
        """
        Args:
            K_osc (float): Kerr of oscillator.
            T1_osc (float): T1 relaxation time of oscillator (seconds).
        """
        self._K_osc = K_osc
        self._T1_osc = T1_osc
        super().__init__(self, *args, **kwargs)

    def _define_fixed_operators(self, N):
        self.I = identity(N)
        self.a = destroy(N)
        self.a_dag = create(N)
        self.q = position(N)
        self.p = momentum(N)
        self.n = num(N)

    @property
    def _hamiltonian(self):
        return -1 / 2 * (2 * pi) * self._K_osc * self.n * self.n  # Kerr

    @property
    def _collapse_operators(self):
        photon_loss = (
            tf.cast(tf.sqrt(1/self._T1_osc), dtype=tf.complex64)
            * self.a
        )

        return [photon_loss]
