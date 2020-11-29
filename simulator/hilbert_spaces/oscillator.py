"""
Simulator for an oscillator (N) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
from math import pi, sqrt
import tensorflow as tf
from tensorflow.keras.backend import batch_dot
from simulator.utils import normalize
from simulator import operators as ops
from .base import HilbertSpace

class Oscillator(HilbertSpace):
    """ Hilbert space of a single oscillator truncated at <N> levels."""
    
    def __init__(self, *args, K_osc, T1_osc, N=100, **kwargs):
        """
        Args:
            K_osc (float): Kerr of the oscillator [Hz].
            T1_osc (float): T1 relaxation time of oscillator [seconds].
            N (int, optional): Size of oscillator Hilbert space. Defaults to 100.
        """
        self._N = N
        self.K_osc = K_osc
        self.T1_osc = T1_osc
        super().__init__(self, *args, **kwargs)

    def _define_operators(self):
        N = self.N
        self.I = ops.identity(N)
        self.a = ops.destroy(N)
        self.q = ops.position(N)
        self.p = ops.momentum(N)
        self.n = ops.num(N)
        self.parity = ops.parity(N)
        self.snap = ops.SNAP(N)
        self.displace = ops.DisplacementOperator(N)
        self.translate = ops.TranslationOperator(N)

    @property
    def _hamiltonian(self):
        n = self.n.to_dense()
        Kerr = -0.5*(2*pi)*self.K_osc * n * n # valid because n is diagonal
        H = tf.linalg.LinearOperatorFullMatrix(Kerr, name='Hamiltonian')
        return H

    @property
    def _dissipator(self):
        a = self.a.to_dense()
        photon_loss = tf.linalg.LinearOperatorFullMatrix(
            sqrt(1/self.T1_osc) * a, name='photon_loss')
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

        psi = normalize(psi)
        return self.measure(psi, Kraus, sample)

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return False