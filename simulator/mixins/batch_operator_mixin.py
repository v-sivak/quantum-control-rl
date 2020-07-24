import tensorflow as tf

from simulator.utils import matrix_flatten


class BatchOperatorMixin:
    """
    Mixin which defines batched operators on a given Hilbert space.
    
    All of these functions are defined so as to be compatible with @tf.function. The
    batch_size is implicit in the shape of the input argument.
    """

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
    def displace(self, amplitude):
        """Calculates D(amplitude) for a batch of amplitudes

        Args:
            amplitude (Tensor([batch_size], c64)): A batch of batch_size amplitudes

        Returns:
            Tensor([batch_size, N, N], c64): A batch of D(amplitude)
        """
        amplitude = matrix_flatten(tf.cast(amplitude, dtype=tf.complex64))
        return tf.linalg.expm(amplitude * self.a_dag - tf.math.conj(amplitude) * self.a)

    @tf.function
    def translate(self, amplitude):
        """Calculates T(amplitude) = D(amplitude / sqrt(2)) for a batch of amplitudes

        Args:
            amplitude (Tensor([batch_size], c64)): A batch of batch_size amplitudes

        Returns:
            Tensor([batch_size, N, N], c64): A batch of T(amplitude)
        """
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex64))
        return self.displace(tf.cast(amplitude, dtype=tf.complex64) / sqrt2)
