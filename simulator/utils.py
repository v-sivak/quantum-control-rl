import tensorflow as tf


def matrix_flatten(tensor):
    """Takes a tensor of arbitrary shape and returns a "flattened" vector of matrices.

    This is useful to get the correct broadcasting shape for batch operations.

    Args:
        tensor (Tensor([x, y, ...])): A tensor with arbitrary shape

    Returns:
        Tensor([numx * numy * ..., 1, 1]): A "flattened" vector of matrices
    """
    tensor = tf.reshape(tensor, [-1])
    tensor = tf.reshape(tensor, shape=[tensor.shape[0], 1, 1])
    return tensor
