# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:34:40 2020

@author: Vladimir Sivak

Qubit is included in the Hilbert space. Simulation is done with a gate-based 
approach to quantum circuits.
"""
import tensorflow as tf
from gkp.gkp_tf_env.gkp_tf_env import GKP
from gkp.gkp_tf_env import helper_functions as hf
from tf_agents import specs
from simulator.hilbert_spaces import OscillatorQubit
from simulator.utils import measurement
from math import pi
from tensorflow import complex64 as c64

class QuantumCircuit(OscillatorQubit, GKP):
    """
    Universal gate sequence for open-loop unitary control of the oscillator.
    The gate sequence consists of 
        1) qubit rotations in the xy-plane of the Bloch sphere
        2) conditional translations of the oscillator conditioned on qubit
    
    """
    def __init__(
        self,
        *args,
        # Required kwargs
        t_gate,
        # Optional kwargs
        **kwargs):
        """
        Args:
            t_gate (float): Gate time in seconds.
        """
        self.t_gate = tf.constant(t_gate, dtype=tf.float32)
        self.step_duration = self.t_gate
        super().__init__(*args, **kwargs)

    @property
    def _quantum_circuit_spec(self):
        spec = {'alpha' : specs.TensorSpec(shape=[2], dtype=tf.float32),
                'beta'  : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'phi'   : specs.TensorSpec(shape=[2], dtype=tf.float32)}
        return spec

    @tf.function
    def _quantum_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'alpha' : Tensor([batch_size,2], tf.float32),
                          'beta'  : Tensor([batch_size,2], tf.float32),
                          'phi'   : Tensor([batch_size,2], tf.float32))

        Returns: see parent class docs

        """
        # Extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        phi_x = action['phi'][:,0]
        phi_y = action['phi'][:,1]

        # Construct gates
        T, CT, R = {}, {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/2.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])
        Rx = self.rotate_qb_xy(phi_x, tf.zeros_like(phi_x))
        Ry = self.rotate_qb_xy(phi_y, pi/2*tf.ones_like(phi_x))
        
        # Qubit rotation
        psi = tf.linalg.matvec(Rx, psi)
        psi = tf.linalg.matvec(Ry, psi)
        # Oscillator translation
        psi = tf.linalg.matvec(T['a'], psi)
        # Conditional translation
        psi = tf.linalg.matvec(CT['b'], psi)

        return psi, psi, tf.ones((self.batch_size,1))

