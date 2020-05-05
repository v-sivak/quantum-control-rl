# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:30:36 2020

@author: Vladimir Sivak
"""
import qutip as qt
import tensorflow as tf
from gkp.gkp_tf_env.gkp_tf_env import GKP
from tensorflow import complex64 as c64

class OscillatorQubitGKP(GKP):
    
    def __init__(self, **kwargs):
        self.tensorstate = True
        super(OscillatorQubitGKP, self).__init__(**kwargs)

    def define_operators(self):
        """
        Define all relevant operators as tensorflow tensors of shape [2N,2N].
        We adopt the notation in which qt.basis(2,0) is a ground state
        Methods need to take care of batch dimension explicitly. 
        
        """
        N = self.N
        # Create qutip tensors
        a = qt.tensor(qt.identity(2), qt.destroy(N))
        a_dag = qt.tensor(qt.identity(2), qt.create(N))
        q = (a.dag() + a) / sqrt(2)
        p = 1j*(a.dag() - a) / sqrt(2)
        
        sz = qt.tensor(qt.sigmaz(), qt.identity(N))
        sx = qt.tensor(qt.sigmax(), qt.identity(N))
        sm = qt.tensor(qt.sigmap(), qt.identity(N))
        I = qt.tensor(qt.identity(2), qt.identity(N))
        hadamard = qt.tensor(qt.qip.operations.snot(), qt.identity(N))
        
        P = [qt.tensor(qt.ket2dm(qt.basis(2,0)), qt.identity(N)), 
             qt.tensor(qt.ket2dm(qt.basis(2,1)), qt.identity(N))] # TODO: do i need this projector?
        
        # Convert to tensorflow tensors
        self.a = tf.constant(a.full(), dtype=c64)
        self.a_dag = tf.constant(a_dag.full(), dtype=c64)
        self.q = tf.constant(q.full(), dtype=c64)
        self.p = tf.constant(p.full(), dtype=c64)
        self.sz = tf.constant(sz.full(), dtype=c64)
        self.sx = tf.constant(sx.full(), dtype=c64)
        self.sm = tf.constant(sm.full(), dtype=c64)
        self.I = tf.constant(I.full(), dtype=c64)
        self.hadamard = tf.constant(hadamard.full(), dtype=c64)
        
        self.P = [tf.constant(P[i].full(), dtype=c64) for i in [0,1]]
        
        # TODO: define Hamiltonian and c_ops here