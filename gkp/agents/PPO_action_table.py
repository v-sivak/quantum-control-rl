# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:36:54 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.policies import greedy_policy

def binary_encoding(x, H):
    """
    Convert x to reversed binary representation with H digits.
    
    """
    code = []
    for _ in range(H):
        code.append(x % 2)
        x = x//2
    return tf.cast(code, tf.float32)


def binary_decoding(x, H):
    """
    Decode reversed binary representation of x with H digits.
    
    """
    z = tf.math.pow(2, tf.range(H))
    x = tf.cast(x, tf.int32)
    return tf.math.reduce_sum(x*z, axis=1)


def convert_policy_to_action_table(policy, H, T):
    """
    Assumes that observations coming from the environment are wrapped into
    vectors of length T + H, where H is horizon length and T is clock period.
    These vectors are used as keys of the hash table. First T components of 
    the key give the one-hot encoding of the clock, and the last T components
    can take values from {-1,1}. Thus, the total number of keys is T*2^H. 
    Keys are converted to table indices using hashfunc(). 
    
    The values in the hash table are vector-actions which need to be converted
    to dictionary format expected by GKP environment. So this is supposed to 
    go with the top-level wrapper of the environment.
    
    Input:
        policy -- policy that needs to be converted to a table representation
        H -- horizon for history of measurements
        T -- clock period (one-hot encoding size)
        
    Output:
        action_table -- tensor of shape=[T*2^H, A], where A is action dim
        
    """   
    keys = []
    for t in range(T):
        for m in range(2**H):
            obs = tf.concat([tf.one_hot(t,T), 2*binary_encoding(m, H)-1], 0)
            keys.append(tf.reshape(obs, shape=[1,T+H]))
    keys = tf.concat(keys, axis=0)  # shape=[T*2^H, T+H]
    ind = hashfunc(keys, H, T)      # shape=[T*2^H]
    # It's already sorted correctly due to the loop structure, but anyways
    keys = tf.gather_nd(keys, ind[:,None])
    
    time_step = ts.transition(keys, tf.zeros(T*2**H))
    action_table = policy.action(time_step).action
    return action_table

    
def hashfunc(key, H, T):
    """
    Hash function to compute the index of 'key' in the action table.
    Expects first T elements of the key to be one-hot encoding of the clock, 
    and last H elements to be the measurement outcomes from {-1,1}.
    
    Input:
        key -- batched tensor of shape [B,T+H]
        H -- horizon for history of measurements
        T -- clock period (one-hot encoding size)
    
    Output:
        index -- tensor of shape [B]
    
    """
    t = tf.argmax(key[:,:T], axis=1, output_type=tf.int32)
    m = binary_decoding((key[:,T:]+1)/2, H)
    return t*2**H + m
    

def action_table_lookup(table, key, H, T):
    """
    Look up the value for the key (observation) in the table.
    
    """
    ind = hashfunc(key, H, T)
    actions = tf.gather_nd(table, ind[:,None])
    return actions


class ActionTablePolicyWrapper(tf_policy.Base):
    """
    Wrapper for the neural net policy that allows to convert it to a table. 
    
    """
    def __init__(self, policy, env):
        """
        Policy is supposed to map input batched tensors to output batched
        tensors. For example, this can be a neural net policy produced by 
        saved_model.load 
        
        Environment needs to be wrapped by ActionWrapper and 
        FlattenObservationsWrapperTF so that it produces observations as 
        batched tensors and admits batched tensor actions (instead of dict 
        observations and actions as in the base GKP class).
        
        """
        super(ActionTablePolicyWrapper, self).__init__(
            env.time_step_spec(), env.action_spec())
        self.make_table(policy, env.H, env.T)

    def _action(self, time_step, policy_state, seed):
        obs = time_step.observation
        action = self.table_lookup(obs)
        return policy_step.PolicyStep(action)

    def make_table(self, policy, H, T):
        action_table = convert_policy_to_action_table(policy, H, T)
        self.table_lookup = lambda key: action_table_lookup(
            action_table, key, H, T)


class ActionTableEpisodeDriver(dynamic_episode_driver.DynamicEpisodeDriver):
    """
    This driver is the same as the standard DynamicEpisodeDriver except that
    it wraps the policy into a table before collecting the episodes. 
    
    Since the PPO collection policy is stochastic, the action table will be 
    sampled from the distribution of policies, but this will happen before 
    any episodes are collected. This trick eliminates the PPO action sampling 
    noise DURING the episode.
    
    This is motivated by the way it will have to be done in a real experiment:
    table is constructed beforehand and off-loaded to FPGA.
    
    """
    def __init__(self, env, policy, observers=None, num_episodes=1,
                 minibatch_size=100):
        self.I = round(num_episodes/minibatch_size)
        policy = ActionTableStochasticPolicyWrapper(policy, env)
        super(ActionTableEpisodeDriver, self).__init__(
            env, policy, observers=observers, num_episodes=minibatch_size)
        
    def run(self):
        self.policy.make_table()
        # Loop to randomize episode durations
        for i in range(self.I):
            super(ActionTableEpisodeDriver, self).run()



class ActionTableStochasticPolicyWrapper(tf_policy.Base):
    """
    Samples a deterministic action table from the stochastic collect policy at 
    the beginning of each episode, and uses this table to map observations to 
    actions during the episode instead of sampling actions from the stochastic
    policy. 

    The implementation is based on the example of greedy_policy.GreedyPolicy.
    This wrapper can be used during the PPO training, i.e. it returns policy
    info needed during training. 
    
    """
    def __init__(self, policy, env, name=None):
        super(ActionTableStochasticPolicyWrapper, self).__init__(
            policy.time_step_spec,
            policy.action_spec,
            policy.policy_state_spec,
            policy.info_spec,
            emit_log_probability=policy.emit_log_probability,
            name=name)
        self._wrapped_policy = policy
        self.H = env.H
        self.T = env.T
    
    @property
    def wrapped_policy(self):
        return self._wrapped_policy
    
    def _variables(self):
        return self._wrapped_policy.variables()

    def make_table(self, policy=None):
        policy = self.wrapped_policy if policy==None else policy
        action_table = convert_policy_to_action_table(policy, self.H, self.T)
        self.table_lookup = lambda key: action_table_lookup(
            action_table, key, self.H, self.T)

    def deterministic_action_distribution(self, time_step):
        """
        Produce a deterministic tfp.distribution centered on the action
        from the current table.
        
        """
        obs = time_step.observation
        action = self.table_lookup(obs)
        return greedy_policy.DeterministicWithLogProb(loc=action)
    
    def _distribution(self, time_step, policy_state):
        """ 
        Returns policy step in the format compaticle with PPO training.
        Action is taken from the current table, and distribution_step.info
        is taken from the wrapped policy (it is a dictionary with keys 'loc' 
        and 'scale' which PPOAgent uses to reconstruct the distribution).
        
        """
        distribution_step = self._wrapped_policy.distribution(
            time_step, policy_state)
        return policy_step.PolicyStep(
            self.deterministic_action_distribution(time_step),
            distribution_step.state, distribution_step.info)
