"""
Simulator for an Oscillator (N) x Qubit (2) Hilbert space

Created on Tue Aug 04 16:08:01 2020

@author: Henry Liu
"""
import qutip as qt
import tensorflow as tf
from numpy import pi, sqrt
from tensorflow import complex64 as c64

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

    def __init__(self, *args, K_osc, T1_osc, T1_qb, T2_qb, **kwargs):
        """
        Args:
            K_osc (float): Kerr of oscillator (Hz).
            T1_osc (float): T1 relaxation time of oscillator (seconds).
            T1_qb (float): T1 relaxation time of qubit (seconds).
            T2_qb (float): T2 decoherence time of qubit (seconds).
        """
        self._K_osc = K_osc
        self._T1_osc = T1_osc
        self._T1_qb = T1_qb
        self._T2_qb = T2_qb

        self._T2_star_qb = 1 / (1 / T2_qb - 1 / (2 * T1_qb))  
        super().__init__(self, *args, **kwargs)

    def _define_fixed_operators(self, N):
        # TODO: Convert this to TensorFlow? #

        # Create qutip tensor ops acting on oscillator Hilbert space
        I = qt.tensor(qt.identity(2), qt.identity(N))
        a = qt.tensor(qt.identity(2), qt.destroy(N))
        a_dag = qt.tensor(qt.identity(2), qt.create(N))
        q = (a.dag() + a) / sqrt(2)
        p = 1j * (a.dag() - a) / sqrt(2)
        n = qt.tensor(qt.identity(2), qt.num(N))

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
