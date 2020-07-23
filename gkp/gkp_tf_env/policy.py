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
from tf_agents.specs import tensor_spec

from tensorflow_core._api.v2.math import real, imag

__all__ = ['IdlePolicy', 'ScriptedPolicyV1', 'ScriptedPolicyV2',
           'SupervisedNeuralNet']
    
    

class IdlePolicy(fixed_policy.FixedPolicy):
    """
    Do nothing policy (zero on all actuators).
    
    """ 
    def __init__(self, time_step_spec, action_spec):
        zero_action = tensor_spec.zero_spec_nest(action_spec)
        super(IdlePolicy, self).__init__(zero_action, 
                                         time_step_spec, action_spec)



class ScriptedPolicy(tf_policy.Base):
    """
    Policy that follows script of actions.
    
    Actions are parametrized according to different gates in the quantum
    circuit executed by the agent at each time step. Action components 
    include 'alpha', 'beta', 'phi' and for certain type of circuit 'epsilon'.
    
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
        self.script = action_script.script # load the script of actions

        # Calculate specs and call init of parent class
        self.dims_map = {'alpha' : 2, 'beta' : 2, 'epsilon' : 2, 'phi' : 1}
        spec = lambda x: specs.TensorSpec(shape=[x], dtype=tf.float32)
        action_spec = {a : spec(self.dims_map[a]) for a in self.script.keys()}        
        policy_state_spec = specs.TensorSpec(shape=[], dtype=tf.int32)

        for a, val in self.script.items():
            if self.dims_map[a] == 2: 
                A = tf.stack([real(val), imag(val)], axis=-1)
            elif self.dims_map[a] == 1:
                A = tf.constant(val, shape=[self.period,1])
            self.script[a] = tf.cast(A, tf.float32)
                
        super(ScriptedPolicy, self).__init__(time_step_spec, action_spec,
                                              policy_state_spec,
                                              automatic_state_reset=True)
        self._policy_info = ()


    def _action(self, time_step, policy_state, seed):
        i = policy_state[0] % self.period # position within the policy period
        out_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        action = {} 
        for a in self.script:
            A = common.replicate(self.script[a][i], out_shape)
            if a == 'alpha': # do Markovian feedback
                A *= time_step.observation['msmt'][:,-1,:]
                if policy_state[0] == 0: A *= 0
            action[a] = A         

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
        

