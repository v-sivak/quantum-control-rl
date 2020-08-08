# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:30:36 2020

@author: Vladimir Sivak
"""
import qutip as qt
import tensorflow as tf
from numpy import sqrt, pi
from tensorflow import complex64 as c64
from tensorflow.keras.backend import batch_dot

from gkp.gkp_tf_env.gkp_tf_env import GKP
from simulator.quantum_trajectory_sim import QuantumTrajectorySim


class OscillatorQubitGKP(GKP):
    """
    This class inherits simulation-independent functionality from the GKP
    class and implements simulation by including the qubit in the Hilbert
    space and using gate-based approach to quantum circuits.
    
    """
    def __init__(self, **kwargs):
        self.tensorstate = True
        super(OscillatorQubitGKP, self).__init__(**kwargs)


    def setup_simulator(self):
        """
        Define all relevant operators as tensorflow tensors of shape [2N,2N].
        We adopt the notation in which qt.basis(2,0) is a qubit ground state.
        Methods need to take care of batch dimension explicitly. 

        Initialize tensorflow quantum trajectory simulator. This is used to
        simulate decoherence, dephasing, Kerr etc using quantum jumps.
        
        """
        N = self.N
        
        # Create qutip tensor ops acting on oscillator Hilbert space
        I = qt.tensor(qt.identity(2), qt.identity(N))
        a = qt.tensor(qt.identity(2), qt.destroy(N))
        a_dag = qt.tensor(qt.identity(2), qt.create(N))
        q = (a.dag() + a) / sqrt(2)
        p = 1j*(a.dag() - a) / sqrt(2)
        n = qt.tensor(qt.identity(2), qt.num(N))
        
        sz = qt.tensor(qt.sigmaz(), qt.identity(N))
        sx = qt.tensor(qt.sigmax(), qt.identity(N))
        sm = qt.tensor(qt.sigmap(), qt.identity(N))
        rxp = qt.tensor(qt.qip.operations.rx(+pi/2), qt.identity(N))
        rxm = qt.tensor(qt.qip.operations.rx(-pi/2), qt.identity(N))
        hadamard = qt.tensor(qt.qip.operations.snot(), qt.identity(N))
        
        # measurement projector
        P = {0 : qt.tensor(qt.ket2dm(qt.basis(2,0)), qt.identity(N)),
             1 : qt.tensor(qt.ket2dm(qt.basis(2,1)), qt.identity(N))}

        # Create qutip Hamiltonian and collapse ops
        Hamiltonian = -1/2 * (2*pi) * self.K_osc * n * n
        
        c_ops = [sqrt(1/self.T1_osc)*a,       # photon loss
                 sqrt(1/self.T1_qb)*sm,       # qubit decay
                 sqrt(0.5/self.Tphi_qb)*sz]   # qubit dephasing
        
        # Convert to tensorflow tensors
        self.I = tf.constant(I.full(), dtype=c64)
        self.a = tf.constant(a.full(), dtype=c64)
        self.a_dag = tf.constant(a_dag.full(), dtype=c64)
        self.q = tf.constant(q.full(), dtype=c64)
        self.p = tf.constant(p.full(), dtype=c64)
        self.n = tf.constant(n.full(), dtype=c64)
        self.sz = tf.constant(sz.full(), dtype=c64)
        self.sx = tf.constant(sx.full(), dtype=c64)
        self.sm = tf.constant(sm.full(), dtype=c64)
        self.rxp = tf.constant(rxp.full(), dtype=c64)
        self.rxm = tf.constant(rxm.full(), dtype=c64)
        self.hadamard = tf.constant(hadamard.full(), dtype=c64)
        
        P = {i : tf.constant(P[i].full(), dtype=c64) for i in [0,1]}
        self.P = {i : tf.stack([P[i]]*self.batch_size) for i in [0,1]}
        
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

        self.mcsteps_read = tf.constant(int(self.t_read / 2 / dt))
        self.mcsteps_gate = tf.constant(int((self.t_gate) / dt))
        self.mcsteps_delay = tf.constant(int(self.t_delay / dt))

        
    @tf.function
    def quantum_circuit_v1(self, psi, action):
        """
        Apply sequenct of quantum gates version 1. In this version conditional 
        translation by 'beta' is not symmetric (translates if qubit is in '1') 
        
        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'phi'
            
        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)
            
        """
        # Extract parameters
        alpha = self.vec_to_complex(action['alpha'])
        beta = self.vec_to_complex(action['beta'])
        phi = action['phi']
        
        # Construct gates
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        sx = tf.stack([self.sx]*self.batch_size)
        I = tf.stack([self.I]*self.batch_size)
        Phase = self.ctrl(I, self.phase(phi)*I)
        T, CT = {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/2.0)
        CT['b'] = self.ctrl(I, T['b'])

        # Feedback translation
        psi_cached = batch_dot(T['a'], psi)
        # Qubit gates
        psi = batch_dot(Hadamard, psi_cached)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.mcsim.run(psi, self.mcsteps_gate)
        psi = batch_dot(CT['b'], psi)
        # Qubit gates
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        # Readout of finite duration
        psi = self.mcsim.run(psi, self.mcsteps_read)
        psi, obs = self.measurement(psi, self.P, sample=True)
        psi = self.mcsim.run(psi,self.mcsteps_read)
        # Feedback delay
        psi = self.mcsim.run(psi, self.mcsteps_delay)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((obs==1), c64) \
            + batch_dot(sx, psi) * tf.cast((obs==-1), c64)

        return psi_final, psi_cached, obs

    @tf.function
    def quantum_circuit_v2(self, psi, action):
        """
        Apply sequenct of quantum gates version 2. In this version conditional 
        translation by 'beta' is symmetric (translates by +-beta/2 controlled 
        by the qubit)
        
        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'phi'
            
        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)
            
        """
        # Extract parameters
        alpha = self.vec_to_complex(action['alpha'])
        beta = self.vec_to_complex(action['beta'])
        phi = action['phi']

        # Construct gates
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        sx = tf.stack([self.sx]*self.batch_size)
        I = tf.stack([self.I]*self.batch_size)
        Phase = self.ctrl(I, self.phase(phi)*I)
        Rotation = self.rotate(action['theta'])
        T, CT = {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/4.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])

        # Feedback translation
        psi = batch_dot(T['a'], psi)
        psi_cached = batch_dot(Rotation, psi)
        # Qubit gates
        psi = batch_dot(Hadamard, psi_cached)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.mcsim.run(psi, self.mcsteps_gate)
        psi = batch_dot(CT['b'], psi)
        # Qubit gates
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        # Readout of finite duration
        psi = self.mcsim.run(psi, self.mcsteps_read)
        psi, obs = self.measurement(psi, self.P, sample=True)
        psi = self.mcsim.run(psi, self.mcsteps_read)
        # Feedback delay
        psi = self.mcsim.run(psi, self.mcsteps_delay)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((obs==1), c64) \
            + batch_dot(sx, psi) * tf.cast((obs==-1), c64)

        return psi_final, psi_cached, obs


    @tf.function
    def quantum_circuit_v3(self, psi, action):
        """
        Apply sequenct of quantum gates version 3. This is a protocol proposed 
        by Baptiste. It essentially combines trimming and sharpening in a 
        single round. Trimming is controlled by 'epsilon'.
        
        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'epsilon', 'phi'
            
        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)
            
        """
        # extract parameters
        alpha = self.vec_to_complex(action['alpha'])
        beta = self.vec_to_complex(action['beta'])
        epsilon = self.vec_to_complex(action['epsilon'])
        phi = action['phi']

        # Construct gates
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        sx = tf.stack([self.sx]*self.batch_size)
        I = tf.stack([self.I]*self.batch_size)
        Phase = self.ctrl(I, self.phase(phi)*I)
        Rxp = tf.stack([self.rxp]*self.batch_size)
        Rxm = tf.stack([self.rxm]*self.batch_size)
        T, CT = {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/4.0)
        T['e'] = self.translate(epsilon/2.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])
        CT['e'] = self.ctrl(tf.linalg.adjoint(T['e']), T['e'])

        # Feedback translation
        psi_cached = batch_dot(T['a'], psi)
        # Qubit gates
        psi = batch_dot(Hadamard, psi_cached)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.mcsim.run(psi, self.mcsteps_gate)
        psi = batch_dot(CT['b'], psi)
        # Qubit rotation
        psi = batch_dot(Rxp, psi)
        # Conditional translation
        psi = batch_dot(CT['e'], psi)
        # Qubit rotation
        psi = batch_dot(Rxm, psi)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.mcsim.run(psi, self.mcsteps_gate)
        psi = batch_dot(CT['b'], psi)
        # Qubit gates
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        # Readout of finite duration
        psi = self.mcsim.run(psi, self.mcsteps_read)
        psi, obs = self.measurement(psi, self.P, sample=True)
        psi = self.mcsim.run(psi, self.mcsteps_read)
        # Feedback delay
        psi = self.mcsim.run(psi, self.mcsteps_delay)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((obs==1), c64) \
            + batch_dot(sx, psi) * tf.cast((obs==-1), c64)

        return psi_final, psi_cached, obs


    @tf.function
    def quantum_circuit_v4(self, psi, action):
        """
        Apply sequence of quantum gates version 4. This is a protocol proposed 
        by Baptiste. It essentially combines trimming and sharpening in a 
        single round. Trimming is controlled by 'epsilon'. This is similar to 
        'v3', but the last conditional displacement gate is replaced with 
        classicaly conditioned feedback.
        
        Input:
            action -- dictionary of batched actions. Dictionary keys are
                      'alpha', 'beta', 'epsilon'
            
        Output:
            psi_final -- batch of final states; shape=[batch_size,N]
            psi_cached -- batch of cached states; shape=[batch_size,N]
            obs -- measurement outcomes; shape=(batch_size,)
            
        """
        # extract parameters
        alpha = self.vec_to_complex(action['alpha'])
        beta = self.vec_to_complex(action['beta'])
        epsilon = self.vec_to_complex(action['epsilon'])

        # Construct gates
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        sx = tf.stack([self.sx]*self.batch_size)
        I = tf.stack([self.I]*self.batch_size)
        Rxp = tf.stack([self.rxp]*self.batch_size)
        Rxm = tf.stack([self.rxm]*self.batch_size)
        T, CT = {}, {}
        T['a'] = self.translate(alpha)
        T['b'] = self.translate(beta/4.0)
        T['e'] = self.translate(epsilon/2.0)
        CT['b'] = self.ctrl(tf.linalg.adjoint(T['b']), T['b'])
        CT['e'] = self.ctrl(tf.linalg.adjoint(T['e']), T['e'])

        # Feedback translation
        psi_cached = batch_dot(T['a'], psi)
        # Qubit gates
        psi = batch_dot(Hadamard, psi_cached)
        # Conditional translation
        psi = batch_dot(CT['b'], psi)
        psi = self.mcsim.run(psi, self.mcsteps_gate)
        psi = batch_dot(CT['b'], psi)
        # Qubit rotation
        psi = batch_dot(Rxp, psi)
        # Conditional translation
        psi = batch_dot(CT['e'], psi)
        # Qubit rotation
        psi = batch_dot(Rxm, psi)
        # Readout of finite duration
        psi = self.mcsim.run(psi, self.mcsteps_read)
        psi, obs = self.measurement(psi, self.P, sample=True)
        psi = self.mcsim.run(psi, self.mcsteps_read)
        # Feedback delay
        psi = self.mcsim.run(psi, self.mcsteps_delay)
        # Flip qubit conditioned on the measurement
        psi_final = psi * tf.cast((obs==1), c64) \
            + batch_dot(sx, psi) * tf.cast((obs==-1), c64)

        return psi_final, psi_cached, obs


    @tf.function # TODO: add losses in phase estimation?
    def phase_estimation(self, psi, beta, angle, sample=False):
        """
        One round of phase estimation. 
        
        Input:
            psi -- batch of state vectors; shape=[batch_size,2N]
            beta -- translation amplitude. shape=(batch_size,)
            angle -- angle along which to measure qubit. shape=(batch_size,)
            sample -- bool flag to sample or return expectation value
        
        Output:
            psi -- batch of collapsed states if sample==True, otherwise same 
                   as input psi; shape=[batch_size,2N]
            z -- batch of measurement outcomes if sample==True, otherwise
                 batch of expectation values of qubit sigma_z.
                 
        """
        I = tf.stack([self.I]*self.batch_size)
        CT = self.ctrl(I, self.translate(beta))
        Phase = self.ctrl(I, self.phase(angle)*I)
        Hadamard = tf.stack([self.hadamard]*self.batch_size)
        
        psi = batch_dot(Hadamard, psi)
        psi = batch_dot(CT, psi)
        psi = batch_dot(Phase, psi)
        psi = batch_dot(Hadamard, psi)
        psi = self.normalize(psi)
        return self.measurement(psi, self.P, sample=sample)

    

    @tf.function
    def ctrl(self, U0, U1):
        """
        Batch controlled-U gate. Apply 'U0' if qubit is '0', and 'U1' if 
        qubit is '1'.
        
        Input:
            U0 -- unitary on the oscillator subspace written in the combined 
                  qubit-oscillator Hilbert space; shape=[batch_size,2N,2N]
            U1 -- same as above
                  
        """
        return batch_dot(self.P[0], U0) + batch_dot(self.P[1], U1)
