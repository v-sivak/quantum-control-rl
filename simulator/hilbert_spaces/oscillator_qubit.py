"""
Simulator for an Oscillator (N) x Qubit (2) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
import tensorflow as tf
from numpy import pi, sqrt
from tensorflow import complex64 as c64
from simulator.utils import measurement, tensor
from .base import HilbertSpace
from simulator import operators as ops

class OscillatorQubit(HilbertSpace):
    """
    Define all relevant operators as tensorflow tensors of shape [2N,2N].
    We adopt the notation in which qt.basis(2,0) is a qubit ground state.
    Methods need to take care of batch dimension explicitly.

    Initialize tensorflow quantum trajectory simulator. This is used to
    simulate decoherence, dephasing, Kerr etc using quantum jumps.
    """

    def __init__(self, *args, K_osc, T1_osc, T1_qb, T2_qb, chi_prime, N=100, **kwargs):
        """
        Args:
            K_osc (float): Kerr of oscillator (Hz).
            T1_osc (float): T1 relaxation time of oscillator (seconds).
            T1_qb (float): T1 relaxation time of qubit (seconds).
            T2_qb (float): T2 decoherence time of qubit (seconds).
            N (int, optional): Size of oscillator Hilbert space.
        """
        self._N = N
        self._K_osc = K_osc
        self._T1_osc = T1_osc
        self._T1_qb = T1_qb
        self._T2_qb = T2_qb
        self._chi_prime = chi_prime

        self._T2_star_qb = 1 / (1 / T2_qb - 1 / (2 * T1_qb))  
        super().__init__(self, *args, **kwargs)

    def _define_fixed_operators(self):
        N = self.N
        self.I = tensor([ops.identity(2), ops.identity(N)])
        self.a = tensor([ops.identity(2), ops.destroy(N)])
        self.a_dag = tensor([ops.identity(2), ops.create(N)])
        self.q = tensor([ops.identity(2), ops.position(N)])
        self.p = tensor([ops.identity(2), ops.momentum(N)])
        self.n = tensor([ops.identity(2), ops.num(N)])
        self.parity = tensor([ops.identity(2), ops.parity(N)])

        self.sx = tensor([ops.sigma_x(), ops.identity(N)])
        self.sy = tensor([ops.sigma_y(), ops.identity(N)])
        self.sz = tensor([ops.sigma_z(), ops.identity(N)])
        self.sm = tensor([ops.sigma_m(), ops.identity(N)])
        self.hadamard = tensor([ops.hadamard(), ops.identity(N)])

        tensor_with = [ops.identity(2), None]
        self.phase = ops.Phase()
        self.translate = ops.TranslationOperator(N, tensor_with=tensor_with)
        self.displace = lambda a: self.translate(sqrt(2)*a)
        self.rotate = ops.RotationOperator(N, tensor_with=tensor_with)

        self.SNAP = ops.SNAP(N, tensor_with=tensor_with)        
        tf.random.set_seed(0)
        offset = tf.cast(tf.random.uniform([N], maxval=0.5, seed=0), c64)
        self.SNAP_miscalibrated = ops.SNAPv2(N, phase_offset=offset)

        tensor_with = [None, ops.identity(N)]
        self.rotate_qb_xy = ops.QubitRotationXY(tensor_with=tensor_with)
        self.rotate_qb_z = ops.QubitRotationZ(tensor_with=tensor_with)
        self.rxp = self.rotate_qb_xy(tf.constant(pi/2), tf.constant(0))
        self.rxm = self.rotate_qb_xy(tf.constant(-pi/2), tf.constant(0))
        
        # qubit sigma_z measurement projector
        self.P = {i : tensor([ops.projector(i,2), ops.identity(N)])
                  for i in [0, 1]}

        self.sx_selective = tensor([ops.sigma_x(), ops.projector(0, N)]) + \
            tensor([ops.identity(2), ops.identity(N)-ops.projector(0, N)])

    @property
    def _hamiltonian(self):
        chi_prime = 1/4 * (2*pi) * self._chi_prime * self.ctrl(self.n**2, -self.n**2)
        kerr = - 1/2 * (2*pi) * self._K_osc * self.n ** 2  # Kerr
        return kerr + chi_prime

    @property
    def _collapse_operators(self):
        photon_loss = (
            tf.cast(tf.sqrt(1/self._T1_osc), dtype=tf.complex64)
            * self.a
        )
        qubit_decay = (
            tf.cast(tf.sqrt(1/self._T1_qb), dtype=tf.complex64)
            * self.sm
        )
        qubit_dephasing = (
            tf.cast(tf.sqrt(1/(2*self._T2_star_qb)), dtype=tf.complex64)
            * self.sz
        )

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
        CT = self.ctrl(self.I, U)
        Phase = self.rotate_qb_z(tf.squeeze(angle))

        psi = tf.linalg.matvec(self.hadamard, psi)
        psi = tf.linalg.matvec(CT, psi)
        psi = tf.linalg.matvec(Phase, psi)
        psi = tf.linalg.matvec(self.hadamard, psi)
        return measurement(psi, self.P, sample)

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return True