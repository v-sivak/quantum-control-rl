"""
Simulator for an Oscillator (N) x Qubit (2) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
import tensorflow as tf
from numpy import pi, sqrt
from tensorflow import complex64 as c64
from tensorflow.keras.backend import batch_dot
from simulator.utils import normalize, tensor
import simulator.operators as ops
from .base import HilbertSpace

class OscillatorQubit(HilbertSpace):
    """
    Hilbert space of an oscillator coupled to a qubit. 
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

        snap = ops.SNAP(N) 
        displace = ops.DisplacementOperator(N)
        translate = ops.TranslationOperator(N)
        self.snap = lambda a: tensor([ops.identity(2), snap(a)])
        self.displace = lambda a: tensor([ops.identity(2), displace(a)])
        self.translate = lambda a: tensor([ops.identity(2), translate(a)])

        # qubit sigma_z measurement projector
        self.P = {i : tensor([ops.projector(i,2), ops.identity(N)]) 
                  for i in [0, 1]}

    @property
    def _hamiltonian(self):
        n = self.n.to_dense()                
        Kerr = -0.5*(2*pi)*self.K_osc * n * n # valid because n is diagonal
        H = tf.linalg.LinearOperatorFullMatrix(Kerr, name='Hamiltonian')
        return H

    @property
    def _dissipator(self):
        a = self.a.to_dense()
        sm = self.sm.to_dense()
        sz = self.sz.to_dense()
        
        photon_loss = tf.linalg.LinearOperatorFullMatrix(
            sqrt(1/self.T1_osc) * a, name='photon_loss')
        
        qubit_decay = tf.linalg.LinearOperatorFullMatrix(
            sqrt(1/self.T1_qb) * sm, name='qubit_decay')
        
        qubit_dephasing = tf.linalg.LinearOperatorFullMatrix(
            sqrt(1/(2*self.T2_star_qb)) * sz, name='qubit_dephasing')

        return [photon_loss, qubit_decay, qubit_dephasing]

    @tf.function
    def ctrl(self, U0, U1):
        """
        Batch controlled-U gate. Apply 'U0' if qubit is '0', and 'U1' if
        qubit is '1'.

        Input:
            U0 -- unitary on the oscillator subspace written in the combined
                  qubit-oscillator Hilbert space; shape=[batch_size,2N,2N]
            U1 -- same as above

        """
        return self.P[0] @ U0 + self.P[1] @ U1

    @tf.function  # TODO: add losses in phase estimation?
    def phase_estimation(self, psi, U, angle, sample=False):
        """
        One round of phase estimation.

        Input:
            psi -- batch of state vectors; shape=[batch_size,2N]
            U -- unitary on which to do phase estimation. shape=(batch_size,N,N)
            angle -- angle along which to measure qubit. shape=(batch_size,)
            sample -- bool flag to sample or return expectation value

        Output:
            psi -- batch of collapsed states if sample==True, otherwise same
                   as input psi; shape=[batch_size,2N]
            z -- batch of measurement outcomes if sample==True, otherwise
                 batch of expectation values of qubit sigma_z.

        """
        I = tf.stack([self.I]*self.batch_size)
        CT = self.ctrl(I, U)
        Phase = self.rotate_qb(angle, axis='z')
        Hadamard = tf.stack([self.hadamard]*self.batch_size)

        psi = batch_dot(Hadamard, psi)
        psi = batch_dot(CT, psi)
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        psi = normalize(psi)
        return self.measure(psi, self.P, sample)

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return True
    
