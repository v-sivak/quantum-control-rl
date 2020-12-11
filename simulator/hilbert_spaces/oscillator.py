"""
Simulator for an oscillator (N) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
from math import pi, sqrt
import tensorflow as tf
from tensorflow.keras.backend import batch_dot
from simulator import operators as ops
from simulator.utils import measurement
from .base import HilbertSpace

class Oscillator(HilbertSpace):
    """
    Define all relevant operators as tensorflow tensors of shape [N,N].
    Methods need to take care of batch dimension explicitly.

    Initialize tensorflow quantum trajectory simulator. This is used to
    simulate decoherence, dephasing, Kerr etc using quantum jumps.
    """

    def __init__(self, *args, K_osc, T1_osc, N=100, channel='quantum_jumps', **kwargs):
        """
        Args:
            K_osc (float): Kerr of oscillator.
            T1_osc (float): T1 relaxation time of oscillator (seconds).
            N (int, optional): Size of oscillator Hilbert space. Defaults to 100.
            channel (str, optional): model of the error channel, either 'diffusion'
                     or 'quantum_jumps'.
        """
        self._N = N
        self._K_osc = K_osc
        self._T1_osc = T1_osc
        super().__init__(self, *args, N=N, channel=channel, **kwargs)

    def _define_fixed_operators(self, N):
        self.I = ops.identity(N)
        self.a = ops.destroy(N)
        self.a_dag = ops.create(N)
        self.q = ops.position(N)
        self.p = ops.momentum(N)
        self.n = ops.num(N)
        self.parity = ops.parity(N)
        self.phase = ops.Phase()
        self.SNAP = ops.SNAP(N)
        self.rotate = ops.RotationOperator(N)
        self.translate = ops.TranslationOperator(N)
        self.displace = lambda a: self.translate(sqrt(2)*a)

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

    @tf.function
    def phase_estimation(self, psi, U, angle, sample=False):
        """
        One round of phase estimation.

        Input:
            psi -- batch of state vectors; shape=[batch_size,N]
            U -- unitary on which to do phase estimation. shape=(batch_size,N,N)
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
        Kraus[0] = 1/2*(I + self.phase(angle)*U)
        Kraus[1] = 1/2*(I - self.phase(angle)*U)

        return measurement(psi, Kraus, sample)

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return False