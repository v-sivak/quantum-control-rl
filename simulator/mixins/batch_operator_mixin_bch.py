"""
Mixin for batched operators (phase, translate-BCH, displace-BCH)
on Oscillator Hilbert space

Created on Sun Jul 26 20:55:36 2020

@author: Henry Liu
"""
import tensorflow as tf

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
        # Pre-diagonalize for displace/translate
        p = tf.cast(self.p, dtype=tf.complex64)
        q = tf.cast(self.q, dtype=tf.complex64)

        # We assume self.p and self.q are already created
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)
        (self._eig_n, self._U_n) = tf.linalg.eigh(self.n)

        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)

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
            tf.reshape(amplitude, [amplitude.shape[0], 1]), dtype=tf.complex64
        )

        # Take real/imag of amplitude for the commutator part of the expansion
        re_a = tf.cast(tf.math.real(amplitude), dtype=tf.complex64)
        im_a = tf.cast(tf.math.imag(amplitude), dtype=tf.complex64)

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
            dtype=tf.complex64,
        )

    @tf.function
    def displace(self, amplitude):
        """Calculates D(amplitude) = T(amplitude * sqrt(2)) for a batch of amplitudes

        Args:
            amplitude (Tensor([batch_size], c64)): A batch of batch_size amplitudes

        Returns:
            Tensor([batch_size, N, N], c64): A batch of D(amplitude)
        """
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex64))
        return self.translate(tf.cast(amplitude, dtype=tf.complex64) * sqrt2)

    @tf.function
    def rotate(self, angle):
        """Calculates oscillator rotation matrix for a batch of angles.

        Args:
            angle (Tensor([batch_size], c64)): A batch of rotation angles

        Returns:
            Tensor([batch_size, N, N], c64): A batch of R(angle)
        """
        angle = tf.cast(
            tf.reshape(angle, [angle.shape[0], 1]), dtype=tf.complex64)
        
        expm_n = tf.linalg.diag(tf.math.exp(-1j * angle * self._eig_n))
        
        return tf.cast(
            self._U_n
            @ expm_n
            @ tf.linalg.adjoint(self._U_n),
            dtype=tf.complex64,
        )