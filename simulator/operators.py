# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:38:34 2020

@author: Vladimir Sivak
"""
from math import pi, sqrt
import tensorflow as tf
from tensorflow import complex64 as c64
from simulator.utils import tensor

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) >= "2.2":
    diag = tf.linalg.diag
else:
    import numpy as np
    diag = np.diag  # k=1 option is broken in tf.linalg.diag in TF 2.1 (#35761)



### Constant operators

def sigma_x():
    return tf.constant([[0., 1.], [1., 0.]], dtype=c64)


def sigma_y():
    return tf.constant([[0.j, -1.j], [1.j, 0.j]], dtype=c64)


def sigma_z():
    return tf.constant([[1., 0.], [0., -1.]], dtype=c64)


def sigma_m():
    return tf.constant([[0., 1.], [0., 0.]], dtype=c64)


def hadamard():
    return 1/sqrt(2) * tf.constant([[1., 1.], [1., -1.]], dtype=c64)


def identity(N):
    """Returns an identity operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN identity operator
    """
    return tf.eye(N, dtype=c64)


def destroy(N):
    """Returns a destruction (lowering) operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN creation operator
    """
    a = diag(tf.sqrt(tf.range(1, N, dtype=tf.float32)), k=1)
    return tf.cast(a, dtype=c64)


def create(N):
    """Returns a creation (raising) operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN creation operator
    """
    return tf.linalg.adjoint(destroy(N))


def num(N):
    """Returns the number operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN number operator
    """
    return tf.cast(diag(tf.range(0, N)), dtype=c64)


def position(N):
    """Returns the position operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN position operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=c64))
    a_dag = create(N)
    a = destroy(N)
    return tf.cast((a_dag + a) / sqrt2, dtype=c64)


def momentum(N):
    """Returns the momentum operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN momentum operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=c64))
    a_dag = create(N)
    a = destroy(N)
    return tf.cast(1j * (a_dag - a) / sqrt2, dtype=c64)


def parity(N):
    """Returns the photon number parity operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN photon number parity operator
    """
    pm1 = tf.where(tf.math.floormod(tf.range(N),2)==1, -1, 1)
    return tf.cast(diag(pm1), dtype=c64)


def projector(n, N):
    """
    Returns a projector onto n-th basis state in N-dimensional Hilbert space.
    Args:
        n (int): index of basis vector
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN photon number parity operator
    """
    assert n < N
    return tf.cast(diag(tf.one_hot(n, N)), c64)


### Parametrized operators

class ParametrizedOperator():
    
    def __init__(self, N, tensor_with=None):
        """
        Args:
            N (int): dimension of Hilbert space
            tensor_with (list, LinearOperator): a list of operators to compute
                tensor product. By convention, <None> should be used in place
                of this operator in the list. For example, [identity(2), None] 
                will create operator in the Hilbert space of size 2*N acting
                trivially on the first component in the tensor product.
        """
        self.N = N
        self.tensor_with = tensor_with

    @tf.function
    def __call__(self, *args, **kwargs):
        this_op = self.compute(*args, **kwargs)
        if self.tensor_with is not None:
            ops = [T if T is not None else this_op for T in self.tensor_with]
            return tensor(ops)
        else:
            return this_op

    def compute(self):
        """ To be implemented by the subclass. """
        pass


class HamiltonianEvolutionOperator(ParametrizedOperator):
    """ Unitary evolution according to some Hamiltonian. """    
    def __init__(self, H, *args, **kwargs):
        """
        Args:
            H (Tensor([N, N], c64)): Hamiltonian in Hz
        """
        N = H.shape[-1]
        (self.eigvals, self.U) = tf.linalg.eigh(H)
        super().__init__(N=N, *args, **kwargs)

    def compute(self, t):
        """
        Args:
            t: time in seconds
        """
        t = tf.cast(t, c64)
        exp_diag = tf.linalg.diag(tf.math.exp(-1j * 2*pi * t * self.eigvals))
        return tf.cast(self.U @ exp_diag @ tf.linalg.adjoint(self.U), c64)


class TranslationOperator(ParametrizedOperator):
    """ 
    Translation in phase space.
    
    Example:
        T = TranslationOperator(100)
        alpha = tf.constant([1.23+0.j, 3.56j, 2.12+1.2j])
        T(alpha) # shape=[3,100,100]
    """
    def __init__(self, N, *args, **kwargs):
        """ Pre-diagonalize position and momentum operators."""
        p = momentum(N)
        q = position(N)
        
        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)
        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, amplitude):
        """Calculates T(amplitude) for a batch of amplitudes using BCH.
        Args:
            amplitude (Tensor([B1, ..., Bb], c64)): A batch of amplitudes
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of T(amplitude)
        """
        # Reshape amplitude for broadcast against diagonals
        amplitude = tf.cast(tf.expand_dims(amplitude, -1), dtype=c64)

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


class DisplacementOperator(TranslationOperator):
    """ 
    Displacement in phase space D(amplitude) = T(amplitude * sqrt(2)).
    
    """    
    def __call__(self, amplitude):
        return super().__call__(amplitude*sqrt(2))


class RotationOperator(ParametrizedOperator):
    """ Rotation in phase space."""    

    def compute(self, phase):
        """Calculates R(phase) = e^{i*phase*n} for a batch of phases.
        Args:
            phase (Tensor([B1, ..., Bb], c64)): A batch of phases
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of R(phase)
        """
        phase = tf.squeeze(phase)
        phase = tf.cast(tf.expand_dims(phase, -1), dtype=c64)
        exp_diag = tf.math.exp(1j * phase * tf.cast(tf.range(self.N), c64))
        return tf.linalg.diag(exp_diag)


class SNAP(ParametrizedOperator):
    """
    Selective Number-dependent Arbitrary Phase (SNAP) gate.
    SNAP(theta) = sum_n( e^(i*theta_n) * |n><n| )
    
    """     
    def __init__(self, N, phase_offset=None, *args, **kwargs):
        """
        Args:
            N (int): dimension of Hilbert space    
            phase_offset (Tensor([N], c64)): static offset added to the rota-
                tion phases to model miscalibrated gate.             
        """
        self.phase_offset = 0 if phase_offset is None else phase_offset
        super().__init__(N=N, *args, **kwargs)
     
    def compute(self, theta):
        """Calculates ideal SNAP(theta) for a batch of SNAP parameters.
        Args:
            theta (Tensor([B1, ..., Bb, S], c64)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of SNAP(theta)
        """
        S = theta.shape[-1] # SNAP truncation
        D = len(theta.shape)-1
        paddings = tf.constant([[0,0]]*D + [[0,self.N-S]])
        theta = tf.cast(theta, dtype=c64)
        theta = tf.pad(theta, paddings)
        theta -= self.phase_offset
        exp_diag = tf.math.exp(1j*theta)
        return tf.linalg.diag(exp_diag)
    

class QubitRotationXY(ParametrizedOperator):
    """
    Qubit rotation in xy plane.
    R(angle, phase) = e^(-i*angle/2*[cos(phase)*sx + sin(phase*sy]))
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(N=2,  *args, **kwargs)

    def compute(self, angle, phase):
        """Calculates rotation matrix for a batch of rotation angles.
        Args:
            angle (Tensor([B1, ..., Bb], float32)): batched angle of rotation
                in radians, i.e. angle=pi corresponds to full qubit flip.
            phase (Tensor([B1, ..., Bb], float32)): batched axis of rotation
                in radians, where by convention 0 is x axis.
        Returns:
            Tensor([B1, ..., Bb, 2, 2], c64): A batch of R(angle, phase)
        """
        assert angle.shape == phase.shape
        angle = tf.cast(tf.reshape(angle, angle.shape+[1,1]), c64)
        phase = tf.cast(tf.reshape(phase, phase.shape+[1,1]), c64)
        
        sx = sigma_x()
        sy = sigma_y()
        I = identity(2)
        
        R = tf.math.cos(angle/2) * I - 1j*tf.math.sin(angle/2) * \
            (tf.math.cos(phase)*sx + tf.math.sin(phase)*sy)
        return R

class QubitRotationZ(ParametrizedOperator):
    """ Qubit rotation around z zxis. R(angle) = e^(-i*angle/2*sz)"""
    def __init__(self,  *args, **kwargs):
        super().__init__(N=2, *args, **kwargs)

    def compute(self, angle):
        """Calculates rotation matrix for a batch of rotation angles.
        Args:
            angle (Tensor([B1, ..., Bb], float32)): batched angle of rotation
                in radians, i.e. angle=pi corresponds to full qubit flip.
        Returns:
            Tensor([B1, ..., Bb, 2, 2], c64): A batch of R(angle)
        """
        angle = tf.cast(tf.reshape(angle, angle.shape+[1,1]), c64)
        
        sz = sigma_z()
        I = identity(2)
        
        R = tf.math.cos(angle/2) * I - 1j*tf.math.sin(angle/2) * sz
        return R


class Phase(ParametrizedOperator):
    """ Simple phase factor."""
    def __init__(self, *args, **kwargs):
        super().__init__(N=1, *args, **kwargs)
        
    def compute(self, angle):
        """
        Calculates batch phase factor e^(i*angle)
        Args:
            angle (Tensor([B1, ..., Bb], float32)): batch of angles in radians
            
        Returns:
            Tensor([B1, ..., Bb, 1, 1], c64): A batch of phase factors
        """
        angle = tf.squeeze(angle) # TODO: get rid of this
        angle = tf.cast(tf.reshape(angle, angle.shape+[1,1]), c64)
        return tf.math.exp(1j * angle)


class SNAPv2(ParametrizedOperator):
    """
    Selective Number-dependent Arbitrary Phase (SNAP) gate.
    SNAP(theta) = sum_n( e^(i*theta_n) * |n><n| )
    This implementation allows to create a miscalibrated SNAP.
    
    """
    def __init__(self, N, angle_offset=None, phase_offset=None, *args):
        """
        Args:
            N (int): dimension of Hilbert space    
            angle_offset (Tensor([N], c64)): static offset added to the rota-
                tion angles to model miscalibrated gate.
            phase_offset (Tensor([N], c64)): static offset added to the rota-
                tion phases to model miscalibrated gate.
        """
        self.rotate_qb = QubitRotationXY()
        self.projectors = tf.stack([projector(i,N) for i in range(N)])
        self.projectors = tf.cast(self.projectors, c64)
        self.angle_offset = 0 if angle_offset is None else angle_offset
        self.phase_offset = 0 if phase_offset is None else phase_offset
        super().__init__(N=N, *args)
    
    def compute(self, theta, dangle=None):
        """Calculates SNAP(theta) using qubit rotation gates. Can simulate
        miscalibrated snap if some noise is added to the rotation angles.
        
        Args:
            theta (Tensor([B1, ..., Bb, S], c64)): A batch of parameters.
            dangle (Tensor([B1, ..., Bb, S], c64)): A batch of offsets to
                add to qubit rotation angles to compenstate for possible
                angle offsets due to miscalibration.
        Returns:
            Tensor([B1, ..., Bb, 2N, 2N], c64): A batch of SNAP(theta)
        """
        # this part is the same as for perfect SNAP: pad the angles with zeros
        S = theta.shape[-1] # SNAP truncation
        batch_shape = theta.shape[:-1]
        paddings = tf.constant([[0,0]]*len(batch_shape) + [[0,self.N-S]])
        theta = tf.cast(theta, dtype=c64) # shape=[B,S]
        dangle = tf.zeros_like(theta) if dangle is None else tf.cast(dangle, dtype=c64)
        theta = tf.pad(theta, paddings) # shape=[B,N]
        dangle = tf.pad(dangle, paddings)
        
        # unitary corresponding to the first unselective qubit flip
        unselective_rotation = tensor(
            [self.rotate_qb(tf.constant(pi),tf.constant(0)), identity(self.N)])
        
        # construct a unitary corresponding to second selective qubit pulse
        angle = tf.ones_like(theta) * pi + self.angle_offset + dangle
        phase = pi - theta + self.phase_offset
        R = self.rotate_qb(angle, phase) # shape=[B,N,2,2]
        projectors = tf.broadcast_to(self.projectors, 
                            batch_shape+self.projectors.shape) # shape=[B,N,N,N]
        selective_rotations = tensor([R, projectors]) # shape=[B,N,2N,2N]
        selective_rotations = tf.reduce_sum(selective_rotations, axis=-3) # shape=[B,2N,2N]
        
        snap = selective_rotations @ unselective_rotation
        return snap
