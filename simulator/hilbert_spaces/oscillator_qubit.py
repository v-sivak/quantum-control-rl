"""
Simulator for an Oscillator (N) x Qubit (2) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
import tensorflow as tf
from numpy import pi, sqrt
from tensorflow import complex64 as c64
from tensorflow.keras.backend import batch_dot
from simulator.utils import normalize, tensor, measurement
import simulator.operators as ops
from .base import HilbertSpace

class OscillatorQubit(HilbertSpace):
    """
    Hilbert space of oscillator coupled to qubit. 
    Oscillator is truncated at <N> levels, qubit is 2 levels.
    We adopt the notation in which basis(0,2) is a qubit ground state.
    Qubit goes first in the tensor products.

    """
    def __init__(self, *args, K_osc, T1_osc, T1_qb, T2_qb, N=100, **kwargs):
        """
        Args:
            K_osc (float): Kerr of oscillator (Hz).
            T1_osc (float): T1 relaxation time of oscillator (seconds).
            T1_qb (float): T1 relaxation time of qubit (seconds).
            T2_qb (float): T2 decoherence time of qubit (seconds).
            N (int, optional): Size of oscillator Hilbert space.
        """
        self._N = N
        self.K_osc = K_osc
        self.T1_osc = T1_osc
        self.T1_qb = T1_qb
        self.T2_qb = T2_qb
        self.T2_star_qb = 1 / (1 / T2_qb - 1 / (2 * T1_qb))
        
        super().__init__(self, *args, N=N, **kwargs)

    def _define_operators(self):
        N = self.N
        # Create tensor-product ops acting on the combined Hilbert space
        self.I = tensor([ops.identity(2), ops.identity(N)])
        self.a = tensor([ops.identity(2), ops.destroy(N)])
        self.n = tensor([ops.identity(2), ops.num(N)])
        self.parity = tensor([ops.identity(2), ops.parity(N)])

        self.sx = tensor([ops.sigma_x(), ops.identity(N)])
        self.sy = tensor([ops.sigma_x(), ops.identity(N)])
        self.sz = tensor([ops.sigma_x(), ops.identity(N)])
        self.sm = tensor([ops.sigma_m(), ops.identity(N)])
        self.hadamard = tensor([ops.hadamard(), ops.identity(N)])

        tensor_with = [ops.identity(2), None]
        self.snap = ops.SNAP(N, tensor_with=tensor_with)
        self.phase = ops.Phase(N, tensor_with=tensor_with)
        self.translate = ops.TranslationOperator(N, tensor_with=tensor_with)
        self.displace = ops.DisplacementOperator(N, tensor_with=tensor_with)
        self.rotate = ops.RotationOperator(N, tensor_with=tensor_with)

        # qubit sigma_z measurement projector
        self.P = {i : tensor([ops.projector(i,2), ops.identity(N)])
                  for i in [0, 1]}

    @property
    def _hamiltonian(self):
        Kerr = -0.5*(2*pi)*self.K_osc*self.n*self.n # valid because n is diagonal
        return Kerr

    @property
    def _dissipator(self):
        photon_loss = sqrt(1/self.T1_osc) * self.a
        qubit_decay = sqrt(1/self.T1_qb) * self.sm
        qubit_dephasing = sqrt(1/(2*self.T2_star_qb)) * self.sz
        return [photon_loss, qubit_decay, qubit_dephasing]


    @tf.function
    def ctrl(self, U):
        """
        Batch controlled-U gate: P[0]*U[0] + P[1]*U[1], where P is qubit projector.

        Args:
            U (dict, LinearOperator): dictionary whose values are unitaries
                to be applied to oscillator conditioned on the qubit state.
                Unitaries are given in combined Hilbert space, but it is 
                assumed that they act only on the oscillator subspace.
        """
        return self.P[0] @ U[0] + self.P[1] @ U[1]

    @tf.function  # TODO: add losses in phase estimation?
    def phase_estimation(self, state, U, angle, sample=False):
        """
        Batch phase estimation of unitary operator <U> using ancilla qubit.
        Corresponds to <sigma_z> = Re(U) * cos(angle) - Im(U) * sin(angle).

        Args:
            state (Tensor([B1,...Bb,2*N], c64)): batch of states
            U (LinearOperator): unitary on which to do phase estimation.
            angle (Tensor([B1,...Bb], c64)): angle in radians
            sample (bool): flag to sample or return expectation value

        Returns:
            state (Tensor([B1,...Bb,2*N], c64)): batch of collapsed states if 
                sample=true; duplicate input <state> is sample=false
            z (Tensor([B1,...Bb,1], c64)): batch of measurement outcomes if 
                sample=True; expectation of qubit sigma_z is sample=false
        """

        CT = self.ctrl({0:self.I, 1:U})
        phase = self.ctrl({0:self.I, 1:self.phase(angle)})

        state = self.hadamard.matvec(state)
        state = CT.matvec(state)
        state = phase.matvec(state)
        state = self.hadamard.matvec(state)
        return measurement(state, self.P, sample)

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return True
    
