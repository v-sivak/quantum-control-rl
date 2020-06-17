# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:36:54 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
from tf_agents.trajectories import time_step as ts


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
    be the top-level wrapper of the environment.
    
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
    Look up the value for the key in the table.
    
    """
    ind = hashfunc(key, H, T)
    actions = tf.gather_nd(table, ind[:,None])
    return actions
    

def action_table_collect_driver():
    pass
    