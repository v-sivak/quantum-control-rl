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
from simulator.utils import normalize
from .base import SimulatorHilbertSpace
from simulator.mixins import BatchOperatorMixinBCH

class OscillatorQubit(SimulatorHilbertSpace, BatchOperatorMixinBCH):
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
        # TODO: Convert this to TensorFlow? #

        # Create qutip tensor ops acting on oscillator Hilbert space
        I = qt.tensor(qt.identity(2), qt.identity(N))
        a = qt.tensor(qt.identity(2), qt.destroy(N))
        a_dag = qt.tensor(qt.identity(2), qt.create(N))
        q = (a.dag() + a) / sqrt(2)
        p = 1j * (a.dag() - a) / sqrt(2)
        n = qt.tensor(qt.identity(2), qt.num(N))
        parity = qt.tensor(qt.identity(2), (1j*pi*qt.num(N)).expm())

        sx = qt.tensor(qt.sigmax(), qt.identity(N))
        sy = qt.tensor(qt.sigmay(), qt.identity(N))
        sz = qt.tensor(qt.sigmaz(), qt.identity(N))
        sm = qt.tensor(qt.sigmap(), qt.identity(N))
        rxp = qt.tensor(qt.qip.operations.rx(+pi / 2), qt.identity(N))
        rxm = qt.tensor(qt.qip.operations.rx(-pi / 2), qt.identity(N))
        hadamard = qt.tensor(qt.qip.operations.snot(), qt.identity(N))

        # measurement projector
        P = {
            0: qt.tensor(qt.ket2dm(qt.basis(2, 0)), qt.identity(N)),
            1: qt.tensor(qt.ket2dm(qt.basis(2, 1)), qt.identity(N)),
        }

        # Convert to tensorflow tensors
        self.I = tf.constant(I.full(), dtype=c64)
        self.a = tf.constant(a.full(), dtype=c64)
        self.a_dag = tf.constant(a_dag.full(), dtype=c64)
        self.q = tf.constant(q.full(), dtype=c64)
        self.p = tf.constant(p.full(), dtype=c64)
        self.n = tf.constant(n.full(), dtype=c64)
        self.sx = tf.constant(sx.full(), dtype=c64)
        self.sy = tf.constant(sy.full(), dtype=c64)
        self.sz = tf.constant(sz.full(), dtype=c64)
        self.sm = tf.constant(sm.full(), dtype=c64)
        self.rxp = tf.constant(rxp.full(), dtype=c64)
        self.rxm = tf.constant(rxm.full(), dtype=c64)
        self.hadamard = tf.constant(hadamard.full(), dtype=c64)
        self.parity = tf.constant(parity.full(), dtype=c64)

        self.P = {i: tf.constant(P[i].full(), dtype=c64) for i in [0, 1]}

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
        Phase = self.rotate_qb(angle, axis='z')
        Hadamard = tf.stack([self.hadamard]*self.batch_size)

        psi = batch_dot(Hadamard, psi)
        psi = batch_dot(CT, psi)
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        psi = normalize(psi)
        return self.measure(psi, self.P, sample)

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