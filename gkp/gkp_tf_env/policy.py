# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:30:15 2020

@author: Vladimir Sivak
"""
import os
import numpy as np
import tensorflow as tf
from math import sqrt, pi
from tensorflow import keras
from tf_agents.policies import fixed_policy, tf_policy
from tf_agents.trajectories import policy_step
from tf_agents import specs
from tf_agents.utils import nest_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

__all__ = ['IdlePolicy', 'ScriptedPolicyV1', 'ScriptedPolicyV2',
           'SupervisedNeuralNet']
    
    

class IdlePolicy(fixed_policy.FixedPolicy):
    """
    Do nothing policy (zero on all actuators).
    
    """ 
    def __init__(self, time_step_spec):
        
        action_spec = specs.TensorSpec(shape=[5], dtype=tf.float32)   
        acts_dim = action_spec.shape[0]
        super(IdlePolicy, self).__init__(tf.zeros(acts_dim), 
                                         time_step_spec, action_spec)


class ScriptedPolicyV1(tf_policy.Base):
    """
    Policy that follows script of actions.
    
    Actions are parametrized according to <quantum_circuit_v1> or <..._v2>
    [Re(alpha), Im(alpha), Re(beta), Im(beta), phi]
    
    Policy has its own memory / clock which stores the current round number.
    
    """
    def __init__(self, time_step_spec, action_script):
        """
        Input:
            time_step_spec -- see tf-agents docs
            action_script -- module or class with attributes 'alpha', 'beta',
                             'phi' and 'period'.
            
        """        
        self.period = action_script.period # periodicity of the protocol
        
        # load the script of actions
        self.beta = tf.constant(action_script.beta, 
                                shape=[self.period,1], dtype=tf.complex64)
        self.alpha = tf.constant(action_script.alpha, 
                                 shape=[self.period,1], dtype=tf.complex64)
        self.phi = tf.constant(action_script.phi, shape=[self.period,1])

        action_spec = specs.TensorSpec(shape=[5], dtype=tf.float32)
        policy_state_spec = specs.TensorSpec(shape=[], dtype=tf.int32, 
                                             name='clock')
        super(ScriptedPolicyV1, self).__init__(time_step_spec, action_spec,
                                              policy_state_spec,
                                              automatic_state_reset=True)
        self._policy_info = ()


    def _action(self, time_step, policy_state, seed):
        i = policy_state[0] % self.period # position within the policy period
        outer_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        phi = common.replicate(self.phi[(i+1)%self.period], outer_shape)
        beta = common.replicate(self.beta[(i+1)%self.period], outer_shape)
        m = time_step.observation['msmt'][:,-1,:] # for Markovian feedback
        alpha = self.alpha[i]*tf.cast(m, dtype=tf.complex64)
        alpha = tf.reshape(alpha, beta.shape)
        action = tf.concat([tf.math.real(alpha), tf.math.imag(alpha), 
                            tf.math.real(beta), tf.math.imag(beta), phi], axis=1)
        return policy_step.PolicyStep(action, policy_state+1, self._policy_info)


class ScriptedPolicyV2(tf_policy.Base):
    """
    Policy that follows script of actions.
    
    Actions are parametrized according to <quantum_circuit_v3>:
    [Re(alpha), Im(alpha), Re(beta), Im(beta), Re(eps), Im(eps), phi]
    
    Policy has its own memory / clock which stores the current round number.
    
    """
    def __init__(self, time_step_spec, action_script):
        """
        Input:
            time_step_spec -- see tf-agents docs
            action_script -- module or class with attributes 'alpha', 'beta',
                             'epsilon', 'phi' and 'period'.
            
        """        
        self.period = action_script.period # periodicity of the protocol
        
        # load the script of actions
        self.beta = tf.constant(action_script.beta, 
                                shape=[self.period,1], dtype=tf.complex64)
        self.alpha = tf.constant(action_script.alpha, 
                                 shape=[self.period,1], dtype=tf.complex64)
        self.epsilon = tf.constant(action_script.epsilon, 
                                   shape=[self.period,1], dtype=tf.complex64)
        self.phi = tf.constant(action_script.phi, shape=[self.period,1])

        action_spec = specs.TensorSpec(shape=[7], dtype=tf.float32)
        policy_state_spec = specs.TensorSpec(shape=[], dtype=tf.int32, 
                                             name='clock')
        super(ScriptedPolicyV2, self).__init__(time_step_spec, action_spec,
                                              policy_state_spec,
                                              automatic_state_reset=True)
        self._policy_info = ()


    def _action(self, time_step, policy_state, seed):
        i = policy_state[0] % self.period # position within the policy period
        outer_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        phi = common.replicate(self.phi[(i+1)%self.period], outer_shape)
        eps = common.replicate(self.epsilon[(i+1)%self.period], outer_shape)
        beta = common.replicate(self.beta[(i+1)%self.period], outer_shape)
        m = time_step.observation['msmt'][:,-1,:] # for Markovian feedback
        alpha = self.alpha[i]*tf.cast(m, dtype=tf.complex64)
        alpha = tf.reshape(alpha, beta.shape)
        action = tf.concat([tf.math.real(alpha), tf.math.imag(alpha), 
                            tf.math.real(beta), tf.math.imag(beta), 
                            tf.math.real(eps), tf.math.imag(eps), phi], axis=1)
        return policy_step.PolicyStep(action, policy_state+1, self._policy_info)


class SupervisedNeuralNet(tf_policy.Base):
    """
    The neural network in this policy was trained in a supervised way to 
    reproduce the results of the MarkovianPolicyV2. It was trained on finite 
    chunks of history with horizon H=16 (pre-padded if necessary to fit this 
    size). The environment needs to return observations with same horizon.
    
    """
    def __init__(self, time_step_spec):
        sim_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims'
        # name = 'Benchmarking_HybridMarkovian4Rounds\supervised_lstm\lstm.hdf5'
        name = 'Benchmarking_HybridMarkovian4Rounds\supervised_dnn\dnn.hdf5'
        # name = 'Benchmarking_HybridMarkovian4Rounds\supervised_linear\shallow.hdf5'
        self.model = keras.models.load_model(os.path.join(sim_dir,name))
        action_spec = specs.TensorSpec(shape=[5], dtype=tf.float32)   
        super(SupervisedNeuralNet, self).__init__(time_step_spec, action_spec)
        
    def _action(self, time_step, policy_state, seed):
        stacked_obs = tf.concat([time_step.observation['action'],
                          time_step.observation['msmt']], axis=2)
        action = self.model.predict(stacked_obs)
        action = tf.convert_to_tensor(action)
        return policy_step.PolicyStep(action)
        

