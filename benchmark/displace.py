"""
A few different implementations of displacement operator calculation

Created on Wed Jun 10 12:44:33 2020

@author: Henry Liu
"""
import warnings

import numpy as np
import scipy as sc
import tensorflow as tf

import utils


def gen_displace_scipy(N):
    """Returns a function to calculate displacements using scipy

    Calling this function calculates the common variables and closes over them.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Callable[[int], Tensor([num, N, N])]: Displacement function for dim N
    """
    a = np.diag(np.sqrt(np.arange(1, N, dtype=np.complex64)), k=1)
    a_dag = np.conjugate(a.transpose())
    vexpm = np.vectorize(sc.linalg.expm, signature="(n,n)->(n,n)")

    def displace(alphas):
        """Calculates D(alpha) for a batch of alphas

        Args:
            alphas (Tensor([num])): A batch of num alphas

        Returns:
            Tensor([num, N, N]): A batch of D(alphas)
        """
        # Reshape for broadcasting [num, 1, 1]
        alphas = np.reshape(alphas, [alphas.shape[0], 1, 1])
        return vexpm(alphas * a_dag - np.conj(alphas) * a)

    return displace


def gen_displace(N):
    """Returns a function to calculate displacements using tensorflow expm

    Calling this function calculates the common variables and closes over them.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Callable[[int], Tensor([num, N, N], c64)]: Displacement function for dim N
    """
    a = utils.destroy(N)
    a_dag = utils.create(N)

    def displace(alphas):
        """Calculates D(alpha) for a batch of alphas

        Args:
            alphas (Tensor([num], c64)): A batch of num alphas

        Returns:
            Tensor([num, N, N], c64): A batch of D(alphas)
        """
        # Reshape for broadcasting [num, 1, 1]
        alphas = tf.reshape(alphas, [alphas.shape[0], 1, 1])
        return tf.linalg.expm(alphas * a_dag - tf.math.conj(alphas) * a)

    return displace


def gen_displace_BCH(N):
    """Returns a function to calculate displacements using tensorflow with the
    Baker-Campbell-Hausdorff formula (BCH).

    Using the BCH formula allows us to perform matrix exponentiation *only* on
    diagonal matrices, which removes the complicated `expm` implementation (using the
    scaling/squaring method and Pade approximation) and changes it to an exponential of
    scalar values along the diagonal.

    Although both should, in theory, still have an O(N^3) complexity, the constant
    multiplier is very different, and there may be additional speedups that can be
    applied to the O(N^3) matrix multiplication steps in BCH.

    Calling this function calculates the common variables and closes over them. Since
    we need to diagonalize q and p in this step, this may take significant time. Of
    course, calling the returned displace(alphas) will be much faster in comparison.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Callable[[int], Tensor([num, N, N], c64)]: Displacement function for dim N
    """
    a = utils.destroy(N, dtype=tf.complex128)
    a_dag = utils.create(N, dtype=tf.complex128)

    # Convert raising/lowering to position/momentum
    sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex128))
    q = (a_dag + a) / sqrt2
    p = (a_dag - a) * 1j / sqrt2

    # Diagonalize q and p
    eig_q, U_q = tf.linalg.eigh(q)
    eig_p, U_p = tf.linalg.eigh(p)
    U_q_dag = tf.linalg.adjoint(U_q)
    U_p_dag = tf.linalg.adjoint(U_p)

    # Calculate the commutator numerically. I'm not sure if this is a little strange in
    # theory, but assuming that [q, p] = j causes significant errors...
    comm = tf.linalg.diag_part(q @ p - p @ q)

    def displace(alphas):
        """Calculates D(alpha) for a batch of alphas

        Args:
            alphas (Tensor([num], c64)): A batch of num alphas

        Returns:
            Tensor([num, N, N], c64): A batch of D(alphas)
        """
        # Scale alpha and reshape from broadcast against the diagonals
        alphas = sqrt2 * tf.cast(
            tf.reshape(alphas, [alphas.shape[0], 1]), dtype=tf.complex128
        )

        # Take real/imag of alphas for the commutator part of the expansion
        re_a = tf.cast(tf.math.real(alphas), dtype=tf.complex128)
        im_a = tf.cast(tf.math.imag(alphas), dtype=tf.complex128)

        # Exponentiate diagonal matrices
        expm_q = tf.linalg.diag(tf.math.exp(1j * im_a * eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1j * re_a * eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * comm))

        # Apply Baker-Campbell-Hausdorff
        return tf.cast(
            U_q @ expm_q @ U_q_dag @ U_p @ expm_p @ U_p_dag @ expm_c,
            dtype=tf.complex64,
        )

    return displace


# Define MPI-wrapped BCH if installed
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_nodes = comm.Get_size()

    def gen_displace_MPI_BCH(N):
        """Returns an MPI-wrapped function to calculate displacements using tensorflow
        with the Baker-Campbell-Hausdorff formula (BCH). This can (and should) be
        called by all nodes when running the MPI routine.

        Args:
            N (int): Dimension of Hilbert space

        Returns:
            Callable[[int], Tensor([num, N, N], c64)]: Displacement function for dim N
        """
        f = gen_displace_BCH(N)

        def routine(alphas):
            """Calculates D(alpha) for a batch of alphas. This function takes into
            account the necessary logic for MPI master/worker.

            Args:
                alphas (Tensor([num], c64)): A batch of num alphas

            Returns:
                Tensor([num, N, N], c64): A batch of D(alphas)
            """
            # Split alphas into {num_nodes} batches
            if rank == 0:
                num_alphas = alphas.shape[0]
                my_batchsize = num_alphas // num_nodes
                alphas = np.reshape(alphas, [num_nodes, -1])
            else:
                my_batchsize = None

            # Broadcast size of batch
            my_batchsize = comm.bcast(my_batchsize, root=0)

            # Allocate receive buffer, scatter alphas from master
            my_alphas = np.empty(my_batchsize, dtype=np.complex64)
            comm.Scatter(alphas, my_alphas, root=0)

            # Calculate displacements for each batch
            my_displs = f(my_alphas)

            # Gather displacements from workers
            displacements = None
            if rank == 0:  # Allocate receive buffer
                displacements = np.empty(
                    [num_nodes, my_batchsize, N, N], dtype=np.complex64
                )
            comm.Gather(my_displs, displacements, root=0)

            # "Squeeze" batched displacements into final shape
            if rank == 0:
                displacements = np.reshape(displacements, [num_alphas, N, N])

            return displacements

        return routine


except ImportError:
    pass  # mpi4py not installed


def gen_displace_distribute_BCH(N, strategy=None):
    """Returns a 'multi-GPU' function to calculate displacements using tensorflow
    with the Baker-Campbell-Hausdorff formula (BCH).

    This uses the tf.distribute training API. At the moment, support is very
    experimental, and only the CentralStorageStrategy is appropriate/implemented for
    our use case. Unfortunately it's also synchronous, and seems to add a significant
    amount of overhead. The ParameterServerStrategy looks promising but custom training
    loop support is planned post TF 2.3

    Args:
        N (int): Dimension of Hilbert space
        strategy (tf.distribute.Strategy, optional): Optional override of default strategy.

    Returns:
        Callable[[int], Tensor([num, N, N], c64)]: Displacement function for dim N
    """
    if strategy is None:
        strategy = tf.distribute.experimental.CentralStorageStrategy()

    if strategy.num_replicas_in_sync == 1:
        warnings.warn("Only one logical device configured!")

    with strategy.scope():
        f = gen_displace_BCH(N)

        def routine(alphas):
            """Calculates D(alpha) for a batch of alphas.

            Args:
                alphas (Tensor([num], c64)): A batch of num alphas

            Returns:
                Tensor([num, N, N], c64): A batch of D(alphas)
            """
            # We re-slice alphas into a Dataset, this is fine even if the routine is
            # being called inside of a Dataset loop already
            alphas = tf.data.Dataset.from_tensor_slices(alphas).batch(alphas.shape[0])
            dist_alphas = strategy.experimental_distribute_dataset(alphas)

            for batch in dist_alphas:
                result = strategy.run(f, args=(batch,))

            return tf.concat(result.values, axis=0)

    return routine
