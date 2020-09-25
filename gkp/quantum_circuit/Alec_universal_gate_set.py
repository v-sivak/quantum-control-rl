# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:34:40 2020

@author: Vladimir Sivak

Qubit is included in the Hilbert space. Simulation is done with a gate-based 
approach to quantum circuits.
"""

class QuantumCircuit(OscillatorQubit, GKP):
    """
    This class inherits simulation-independent functionality from the GKP
    class and implements simulation by abstracting away the qubit and using
    Kraus maps formalism to efficiently simulate operations on the oscillator 
    Hilbert space.
    
    Universal gate sequence for open-loop unitary control of the oscillator.
    The gate sequence consists of 
        1) oscillator translations in phase space 
        2) qubit rotations on the Bloch sphere
        3) conditional translations of the oscillator conditioned on qubit
    
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
            t_read (float): Readout time in seconds.
        """
        self.t_gate = tf.constant(t_gate, dtype=tf.float32)
        self.step_duration = self.t_gate
        super().__init__(*args, **kwargs)

    @tf.function
    def _quantum_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'alpha' : Tensor([batch_size,1,2], tf.float32),
                          'beta'  : Tensor([batch_size,1,2], tf.float32),
                          'phi'   : Tensor([batch_size,1,3], tf.float32))

        Returns: see parent class docs

        """
        # Extract parameters
        alpha = hf.vec_to_complex(action['alpha'])
        beta = hf.vec_to_complex(action['beta'])
        phi_x, phi_y, phi_z = tf.transpose(action['phi'])

        # Construct gates
        T, CT, R = {}, {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/4.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])

        R['x'] = self.rotate_qb(phi_x, axis='x')
        R['y'] = self.rotate_qb(phi_y, axis='y')
        R['z'] = self.rotate_qb(phi_z, axis='z')

        # Oscillator translation
        psi_cached = batch_dot(T['a'], psi)
        # Qubit rotation
        psi = batch_dot(R['x'], psi)
        psi = batch_dot(R['y'], psi)
        psi = batch_dot(R['z'], psi)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.simulate(psi, self.t_gate)
        psi = batch_dot(CT['b'], psi)

        return psi, psi, tf.zeros(self.batch_size)
