# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:24:51 2021

@author: Vladimir Sivak
"""
import tensorflow as tf
from gkp.tf_env.tf_env import GKP
from tf_agents import specs
from simulator.hilbert_spaces import OscillatorQubit

class QuantumCircuit(OscillatorQubit, GKP):
    """
    In the open-loop control case, there are no intermediate observations. 
    Therefore, control sequence can be generated fully from time step 0 to T 
    and sent over to the remote client which will test this sequence and return 
    the reward. This way all communication with remote environment can be lumped 
    into the reward circuit.
    
    To ensure compatibility with the rest of the training pipeline, it is 
    convenient to create this empty circuit. The rest of the code will package
    the time-steps in the correct format expected by the tf-agents library.
    
    """
    @property
    def _quantum_circuit_spec(self):
        spec = {'beta'  : specs.TensorSpec(shape=[2], dtype=tf.float32), 
                'phi'   : specs.TensorSpec(shape=[2], dtype=tf.float32)}
        return spec

    @tf.function
    def _quantum_circuit(self, psi, action):
        """
        Args:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'beta'  : Tensor([batch_size,2], tf.float32),
                          'phi'   : Tensor([batch_size,2], tf.float32))

        Returns: see parent class docs

        """
        return psi, psi, tf.ones((self.batch_size,1))