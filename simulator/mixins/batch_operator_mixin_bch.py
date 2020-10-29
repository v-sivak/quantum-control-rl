"""
Mixin for batched operators (phase, translate-BCH, displace-BCH)
on Oscillator Hilbert space

Created on Sun Jul 26 20:55:36 2020

@author: Henry Liu
"""
import tensorflow as tf
from tensorflow import complex64 as c64
from simulator.utils import matrix_flatten


class BatchOperatorMixinBCH:
    """
    Mixin which defines batched operators on a given Hilbert space. The translate and
    displace operators are implemented with the Baker-Campbell-Hausdorff formula.

    All of these functions are defined so as to be compatible with @tf.function. The
    batch_size is implicit in the shape of the input argument.
    """

    def __init__(self, *args, **kwargs):
        """
        Our mixin's __init__ is just to set-up the diagonalized matrices for displace
        and translate. We pass the arguments up the init chain.
        """
        # Ensure correct dtype for inherited operators
        p = tf.cast(self.p, dtype=c64)
        q = tf.cast(self.q, dtype=c64)
        n = tf.cast(self.n, dtype=c64)
        
        try:
            sx = tf.cast(self.__getattribute__('sx'), dtype=c64)
            sy = tf.cast(self.__getattribute__('sy'), dtype=c64)
            sz = tf.cast(self.__getattribute__('sz'), dtype=c64)
        except AttributeError:
            pass

        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)
        (self._eig_n, self._U_n) = tf.linalg.eigh(n)

        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)

        try:
            (self._eig_sx, self._U_sx) = tf.linalg.eigh(sx)
            (self._eig_sy, self._U_sy) = tf.linalg.eigh(sy)
            (self._eig_sz, self._U_sz) = tf.linalg.eigh(sz)
        except NameError:
            pass

        # Continue up the init chain
        super().__init__(*args, **kwargs)

    @tf.function	
    def phase(self, phi):	
        """	
        Batch phase factor.	
        Input:	
            phi -- tensor of shape (batch_size,) or compatible	
        Output:	
            op -- phase factor; shape=[batch_size,1,1]	
        """	
        phi = matrix_flatten(tf.cast(phi, dtype=tf.complex64))	
        return tf.linalg.expm(1j * phi)

    @tf.function
    def translate(self, amplitude):
        """Calculates T(amplitude) for a batch of amplitudes

        Args:
            amplitude (Tensor([batch_size], c64)): A batch of batch_size amplitudes

        Returns:
            Tensor([batch_size, N, N], c64): A batch of T(amplitude)
        """
        # Reshape amplitude for broadcast against diagonals
        amplitude = tf.cast(
            tf.reshape(amplitude, [amplitude.shape[0], 1]), dtype=c64
        )

        # Take real/imag of amplitude for the commutator part of the expansion
        re_a = tf.cast(tf.math.real(amplitude), dtype=c64)
        im_a = tf.cast(tf.math.imag(amplitude), dtype=c64)

        # Exponentiate diagonal matrices
        expm_q = tf.linalg.diag(tf.math.exp(1j * im_a * self._eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1j * re_a * self._eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * self._qp_comm))

        # Apply Baker-Campbell-Hausdorff
        return tf.cast(
            self._U_q
            @ expm_q
            @ tf.linalg.adjoint(self._U_q)
            @ self._U_p
            @ expm_p
            @ tf.linalg.adjoint(self._U_p)
            @ expm_c,
            dtype=c64,
        )

    @tf.function
    def displace(self, amplitude):
        """Calculates D(amplitude) = T(amplitude * sqrt(2)) for a batch of amplitudes

        Args:
            amplitude (Tensor([batch_size], c64)): A batch of batch_size amplitudes

        Returns:
            Tensor([batch_size, N, N], c64): A batch of D(amplitude)
        """
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=c64))
        return self.translate(tf.cast(amplitude, dtype=c64) * sqrt2)

    @tf.function
    def rotate(self, angle):
        """Calculates oscillator rotation matrix for a batch of angles.

        Args:
            angle (Tensor([batch_size], c64)): A batch of rotation angles

        Returns:
            Tensor([batch_size, N, N], c64): A batch of R(angle)
        """
        angle = tf.cast(tf.reshape(angle, [angle.shape[0], 1]), dtype=c64)

        expm_n = tf.linalg.diag(tf.math.exp(-1j * angle * self._eig_n))

        return tf.cast(
            self._U_n @ expm_n @ tf.linalg.adjoint(self._U_n), dtype=c64,
        )

    @tf.function
    def rotate_qb(self, phi, axis):
        """Calculates qubit rotation matrix for a batch of angles.

        Args:
            phi (Tensor([batch_size], c64)): A batch of rotation angles
            axis (str): rotation axis, one of {'x', 'y', 'z'}

        Returns:
            Tensor([batch_size, N, N], c64): A batch of Rx(phi)
        """
        if axis == 'x':
            eig, U = (self._eig_sx, self._U_sx)
        if axis == 'y':
            eig, U = (self._eig_sy, self._U_sy)
        if axis == 'z':
            eig, U = (self._eig_sz, self._U_sz)
        
        phi = tf.cast(tf.reshape(phi, [phi.shape[0], 1]), dtype=c64)
        exp_eig = tf.linalg.diag(tf.math.exp(-1j * phi * eig / 2))

        return tf.cast(
            U @ exp_eig @ tf.linalg.adjoint(U), dtype=c64,
        )
    
    @tf.function
    def SNAP(self, theta):
        """Batch Selective Number-dependent Arbitrary Phase (SNAP) gate.

        Args:
            theta (Tensor([batch_size, S], c64)): batch of SNAP parameters,
                where S is the largest involved Fock state, S < N.

        Returns:
            Tensor([batch_size, N, N], c64): A batch of SNAP(theta)
        """
        theta = tf.cast(theta, dtype=c64)
        theta = tf.pad(theta, tf.constant([[0,0],[0,self.N-theta.shape[1]]]))
        theta_exp_diag = tf.math.exp(theta)
        snap_osc = tf.linalg.diag(theta_exp_diag)
        snap = self.ctrl(snap_osc, snap_osc) if self.tensorstate else snap_osc
        return snap