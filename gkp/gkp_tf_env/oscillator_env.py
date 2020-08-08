# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:30:01 2020

@author: Vladimir Sivak
"""

import qutip as qt
from numpy import pi, sqrt
import tensorflow as tf
from tensorflow import complex64 as c64
from tensorflow.keras.backend import batch_dot

from gkp.gkp_tf_env.gkp_tf_env import GKP
from simulator.quantum_trajectory_sim import QuantumTrajectorySim

class OscillatorGKP(GKP):
    """
    This class inherits simulation-independent functionality from the GKP
    class and implements simulation by abstracting away the qubit and using
    Kraus maps formalism to rather efficiently simulate operations on the
    oscillator Hilbert space. 
    
    """    
    def __init__(self, **kwargs):
        self.tensorstate = False
        super(OscillatorGKP, self).__init__(**kwargs)


    def setup_simulator(self):
        """
        Define all relevant operators as tensorflow tensors of shape [N,N].
        Methods need to take care of batch dimension explicitly. 
        
        Initialize tensorflow quantum trajectory simulator. This is used to
        simulate decoherence, dephasing, Kerr etc using quantum jumps.
        
        """     
        N = self.N
        
        # Create qutip tensor ops acting on oscillator Hilbert space
        I = qt.identity(N)
        a = qt.destroy(N)
        a_dag = qt.create(N)
        q = (a.dag() + a) / sqrt(2)
        p = 1j*(a.dag() - a) / sqrt(2)
        n = qt.num(N)
        
        # Create qutip Hamiltonian and collapse ops
        Hamiltonian = -1/2*(2*pi)*self.K_osc*n*n  # Kerr
        c_ops = [sqrt(1/self.T1_osc)*a]           # photon loss

        # Convert to TensorFlow tensors
        self.I = tf.constant(I.full(), dtype=c64)
        self.a = tf.constant(a.full(), dtype=c64)
        self.a_dag = tf.constant(a_dag.full(), dtype=c64)
        self.q = tf.constant(q.full(), dtype=c64)
        self.p = tf.constant(p.full(), dtype=c64)
        self.n = tf.constant(n.full(), dtype=c64)

        self.Hamiltonian = tf.constant(Hamiltonian.full(), dtype=c64)
        self.c_ops = [tf.constant(op.full(), dtype=c64) for op in c_ops]        

        # Create Kraus ops for free evolution simulator
        Kraus = {}
        dt = self.discrete_step_duration
        Kraus[0] = self.I - 1j*self.Hamiltonian*dt
        for i, c in enumerate(self.c_ops):
            Kraus[i+1] = sqrt(dt) * c
            Kraus[0] -= 1/2 * tf.linalg.matmul(c, c, adjoint_a=True) * dt
        
        # Initialize quantum trajectories simulator
        self.mcsim = QuantumTrajectorySim(Kraus)
            
        self.mcsteps_round = tf.constant(int((self.t_gate + self.t_read)/dt))
        self.mcsteps_delay = tf.constant(int(self.t_delay/dt))


    @tf.function
    def quantum_circuit_v1(self, psi, action):
        """
        Apply Kraus map version 1. In this version conditional translation by
        'beta' is not symmetric (translates if qubit is in '1') 
        
        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'phi'
            
        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)
            
        """
        # extract parameters
        alpha = self.vec_to_complex(action['alpha'])
        beta = self.vec_to_complex(action['beta'])
        phi = action['phi']
        
        Kraus = {}
        T = {'a' : self.translate(alpha),
             'b' : self.translate(beta)}
        I = tf.stack([self.I]*self.batch_size)
        Kraus[0] = 1/2*(I + self.phase(phi)*T['b'])
        Kraus[1] = 1/2*(I - self.phase(phi)*T['b'])
        
        psi = self.mcsim.run(psi, self.mcsteps_delay)
        psi_cached = batch_dot(T['a'], psi)
        psi = self.mcsim.run(psi_cached, self.mcsteps_round)
        psi = self.normalize(psi)
        psi_final, obs = self.measurement(psi, Kraus, sample=True)
        
        return psi_final, psi_cached, obs

    @tf.function
    def quantum_circuit_v2(self, psi, action):
        """
        Apply Kraus map version 2. In this version conditional translation by
        'beta' is symmetric (translates by +-beta/2 controlled by the qubit)
        
        Input:
            action -- batch of actions; shape=[batch_size,5]
            
        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)
            
        """
        # extract parameters
        alpha = self.vec_to_complex(action['alpha'])
        beta = self.vec_to_complex(action['beta'])
        phi = action['phi']
        Rotation = self.rotate(action['theta'])
        
        Kraus = {}
        T = {'a' : self.translate(alpha),
             'b' : self.translate(beta/2.0)}
        Kraus[0] = 1/2*(tf.linalg.adjoint(T['b']) + self.phase(phi)*T['b'])
        Kraus[1] = 1/2*(tf.linalg.adjoint(T['b']) - self.phase(phi)*T['b'])

        psi = self.mcsim.run(psi, self.mcsteps_delay)
        psi = batch_dot(T['a'], psi)
        psi_cached = batch_dot(Rotation, psi)
        psi = self.mcsim.run(psi_cached, self.mcsteps_round)
        psi = self.normalize(psi)
        psi_final, obs = self.measurement(psi, Kraus, sample=True)
        
        return psi_final, psi_cached, obs


    @tf.function
    def quantum_circuit_v3(self, psi, action):
        """
        Apply Kraus map version 3. This is a protocol proposed by Baptiste.
        It essentially combines trimming and sharpening in a single round. 
        Trimming is controlled by 'epsilon'.
        
        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'epsilon', 'phi'
            
        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=[batch_size,1]
            
        """
        # extract parameters
        alpha = self.vec_to_complex(action['alpha'])
        beta = self.vec_to_complex(action['beta'])
        epsilon = self.vec_to_complex(action['epsilon'])
        phi = action['phi']
        
        Kraus = {}
        T = {}
        T['a'] = self.translate(alpha)
        T['+b'] = self.translate(beta/2.0)
        T['-b'] = tf.linalg.adjoint(T['+b'])
        T['+e'] = self.translate(epsilon/2.0)
        T['-e'] = tf.linalg.adjoint(T['+e'])

        
        chunk1 = 1j*batch_dot(T['-b'], batch_dot(T['+e'], T['+b'])) \
                - 1j*batch_dot(T['-b'], batch_dot(T['-e'], T['+b'])) \
                + batch_dot(T['-b'], batch_dot(T['-e'], T['-b'])) \
                + batch_dot(T['-b'], batch_dot(T['+e'], T['-b']))
                    
        chunk2 = 1j*batch_dot(T['+b'], batch_dot(T['-e'], T['-b'])) \
                - 1j*batch_dot(T['+b'], batch_dot(T['+e'], T['-b'])) \
                + batch_dot(T['+b'], batch_dot(T['-e'], T['+b'])) \
                + batch_dot(T['+b'], batch_dot(T['+e'], T['+b']))
        
        Kraus[0] = 1/4*(chunk1 + self.phase(phi)*chunk2)
        Kraus[1] = 1/4*(chunk1 - self.phase(phi)*chunk2)

        psi = self.mcsim.run(psi, self.mcsteps_delay)
        psi_cached = batch_dot(T['a'], psi)
        psi = self.mcsim.run(psi_cached, self.mcsteps_round)
        psi = self.normalize(psi)
        psi_final, obs = self.measurement(psi, Kraus, sample=True)
        
        return psi_final, psi_cached, obs


    @tf.function
    def quantum_circuit_v4(self, psi, action):
        """
        Apply Kraus map version 4. This is a protocol proposed by Baptiste.
        It essentially combines trimming and sharpening in a single round. 
        Trimming is controlled by 'epsilon'. This is similar to 'v3', but
        the last conditional displacement gate is replaced with classicaly
        conditioned feedback.
        
        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'epsilon'
            
        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=[batch_size,1]
            
        """
        # extract parameters
        alpha = self.vec_to_complex(action['alpha'])
        beta = self.vec_to_complex(action['beta'])
        epsilon = self.vec_to_complex(action['epsilon'])
        
        Kraus = {}
        T = {}
        T['a'] = self.translate(alpha)
        T['+b'] = self.translate(beta/2.0)
        T['-b'] = tf.linalg.adjoint(T['+b'])
        T['+e'] = self.translate(epsilon/2.0)
        T['-e'] = tf.linalg.adjoint(T['+e'])

        
        chunk1 = batch_dot(T['-e'], T['-b']) - 1j*batch_dot(T['-e'], T['+b'])
        chunk2 = batch_dot(T['+e'], T['-b']) + 1j*batch_dot(T['+e'], T['+b'])
        
        Kraus[0] = 1/2/sqrt(2)*(chunk1 + chunk2)
        Kraus[1] = 1/2/sqrt(2)*(chunk1 - chunk2)

        psi = self.mcsim.run(psi, self.mcsteps_delay)
        psi_cached = batch_dot(T['a'], psi)
        psi = self.mcsim.run(psi_cached, self.mcsteps_round)
        psi = self.normalize(psi)
        psi_final, obs = self.measurement(psi, Kraus, sample=True)
        
        return psi_final, psi_cached, obs


    @tf.function
    def phase_estimation(self, psi, beta, angle, sample=False):
        """
        One round of phase estimation. 
        
        Input:
            psi -- batch of state vectors; shape=[batch_size,N]
            beta -- translation amplitude. shape=(batch_size,)
            angle -- angle along which to measure qubit. shape=(batch_size,)
            sample -- bool flag to sample or return expectation value
        
        Output:
            psi -- batch of collapsed states if sample==True, otherwise same 
                   as input psi; shape=[batch_size,N]
            z -- batch of measurement outcomes if sample==True, otherwise
                 batch of expectation values of qubit sigma_z.
                 
        """
        Kraus = {}
        I = tf.stack([self.I]*self.batch_size)
        T_b = self.translate(beta)
        Kraus[0] = 1/2*(I + self.phase(angle)*T_b)
        Kraus[1] = 1/2*(I - self.phase(angle)*T_b)
        
        psi = self.normalize(psi)
        return self.measurement(psi, Kraus, sample=sample)
