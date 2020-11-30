"""
Simulator for an oscillator (N) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
from math import pi, sqrt
import tensorflow as tf
from tensorflow.keras.backend import batch_dot
from simulator.utils import normalize, measurement
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
        self.phase = ops.Phase(N)
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
    def phase_estimation(self, state, U, angle, sample=False):
        """
        Batch phase estimation of unitary operator <U> using ancilla qubit.
        Corresponds to <sigma_z> = Re(U) * cos(angle) - Im(U) * sin(angle).

        Args:
            state (Tensor([B1,...Bb,N], c64)): batch of states
            U (LinearOperator): unitary on which to do phase estimation.
            angle (Tensor([B1,...Bb], c64)): angle in radians
            sample (bool): flag to sample or return expectation value

        Returns:
            state (Tensor([B1,...Bb,N], c64)): batch of collapsed states if 
                sample=true; duplicate input <state> is sample=false
            z (Tensor([B1,...Bb,1], c64)): batch of measurement outcomes if 
                sample=True; expectation of qubit sigma_z is sample=false
        """
        I = self.I.to_dense()
        U = U.to_dense()
        phase = self.phase(angle).to_dense()
        
        Kraus = {}
        Kraus[0] = 1/2*(I + phase * U)
        Kraus[1] = 1/2*(I - phase * U)

        Kraus = {i : tf.linalg.LinearOperatorFullMatrix(K)
                 for i, K in Kraus.items()}

        return measurement(state, Kraus, sample)

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return False