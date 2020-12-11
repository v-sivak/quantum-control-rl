"""
Simulator for an Oscillator (N) x Qubit (2) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
import qutip as qt
import tensorflow as tf
from numpy import pi, sqrt
from tensorflow import complex64 as c64
from tensorflow.keras.backend import batch_dot
from simulator.utils_v2 import measurement, tensor
from .base import SimulatorHilbertSpace
from simulator.mixins import BatchOperatorMixinBCH
from simulator import operators_v2 as ops

class OscillatorQubit(SimulatorHilbertSpace):
    """
    Define all relevant operators as tensorflow tensors of shape [2N,2N].
    We adopt the notation in which qt.basis(2,0) is a qubit ground state.
    Methods need to take care of batch dimension explicitly.

    Initialize tensorflow quantum trajectory simulator. This is used to
    simulate decoherence, dephasing, Kerr etc using quantum jumps.
    """

    def __init__(self, *args, K_osc, T1_osc, T1_qb, T2_qb, N=100, channel='quantum_jumps', **kwargs):
        """
        Args:
            K_osc (float): Kerr of oscillator (Hz).
            T1_osc (float): T1 relaxation time of oscillator (seconds).
            T1_qb (float): T1 relaxation time of qubit (seconds).
            T2_qb (float): T2 decoherence time of qubit (seconds).
            N (int, optional): Size of oscillator Hilbert space.
            channel (str, optional): model of the error channel, either 'diffusion'
                    or 'quantum_jumps'.
        """
        self._N = N
        self._K_osc = K_osc
        self._T1_osc = T1_osc
        self._T1_qb = T1_qb
        self._T2_qb = T2_qb

        self._T2_star_qb = 1 / (1 / T2_qb - 1 / (2 * T1_qb))  
        super().__init__(self, *args, N=N, channel=channel, **kwargs)

    def _define_fixed_operators(self, N):
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
        self.SNAP = ops.SNAP(N, tensor_with=tensor_with)
        self.phase = ops.Phase(tensor_with=tensor_with)
        self.translate = ops.TranslationOperator(N, tensor_with=tensor_with)
        self.displace = lambda a: self.translate(sqrt(2)*a)
        self.rotate = ops.RotationOperator(N, tensor_with=tensor_with)

        tensor_with = [None, ops.identity(N)]
        self.rotate_qb_xy = ops.QubitRotationXY(tensor_with=tensor_with)
        self.rotate_qb_z = ops.QubitRotationZ(tensor_with=tensor_with)

        self.rxp = self.rotate_qb_xy(tf.constant(pi/2), tf.constant(0))
        self.rxm = self.rotate_qb_xy(tf.constant(-pi/2), tf.constant(0))
        
        # qubit sigma_z measurement projector
        self.P = {i : tensor([ops.projector(i,2), ops.identity(N)])
                  for i in [0, 1]}

    @property
    def _hamiltonian(self):
        return -1 / 2 * (2 * pi) * self._K_osc * self.n * self.n  # Kerr

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
        I = tf.stack([self.I]*self.batch_size)
        CT = self.ctrl(I, U)
        Phase = self.rotate_qb_z(tf.squeeze(angle))
        Hadamard = tf.stack([self.hadamard]*self.batch_size)

        psi = batch_dot(Hadamard, psi)
        psi = batch_dot(CT, psi)
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        return measurement(psi, self.P, sample)

    @tf.function
    def rotate_qb_xy(self, phi, theta):
        """Calculate qubit rotation matrix for rotation axis in xy-plane.

        Args:
            phi (Tensor([batch_size], c64)): angle between rotation axis and 
                x-axis of the Bloch sphere.
            theta (Tensor([batch_size], c64)): rotation angle.

        Returns:
            Tensor([batch_size, N, N], c64): A batch of R_phi(theta)
        """
        # ensure correct shapes for 'phi' and 'theta'
        phi = tf.cast(tf.reshape(phi, [phi.shape[0], 1, 1]), dtype=c64)
        theta = tf.cast(tf.reshape(phi, [theta.shape[0], 1, 1]), dtype=c64)
        
        I = tf.expand_dims(self.I, axis=0)
        sx = tf.expand_dims(self.sx, axis=0)
        sy = tf.expand_dims(self.sy, axis=0)
        
        R = tf.math.cos(theta/2)*I - 1j*tf.math.sin(theta/2) * \
            (tf.math.cos(phi)*sx + tf.math.sin(phi)*sy)
        return R

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return True