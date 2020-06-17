# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:36:54 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

# TODO: find a way to not use .numpy() in dictionary key
# TODO: maybe this whole hashfunc thing is not necessary 
# TODO: make hashfunc and table_lookup work with batches

def action_table_collect_driver():
    pass

def convert_policy_to_action_table(policy, H, T):
    """
    Assumes that observations coming from the environment are wrapped into
    vectors of length T + H, where H is horizon length and T is clock period.
    These vectors are used as keys of the hash table. First T components of 
    the key give the one-hot encoding of the clock, and the last T components
    can take values from {-1,1}. Thus, the total number of keys is T*2^H. 
    
    The values in the hash table are vector-actions which need to be converted
    to dictionary format expected by GKP environment. So this is supposed to 
    be the top-level wrapper of the environment.
    
    """   
    action_table = {}
    for t in range(T):
        for m in range(2**H):
            obs = tf.concat([tf.one_hot(t,T), 2*binary_encoding(m, H)-1], 0)
            obs = tf.reshape(obs, shape=[1,T+H])
            time_step = ts.transition(obs, [0])
            action_step = policy.action(time_step)
            key = hashfunc(obs, H, T)[0].numpy()
            action_table[key] = action_step.action
    return action_table

    
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


def hashfunc(x, H, T):
    """
    Hash function to compute the index of key 'x' in the action table.
    Expects shape=[B,T+H] with first T elements being one-hot encoding of
    the clock, and last H elements being measurement outcomes from {-1,1}.
    
    """
    t = tf.argmax(x[:,:T], axis=1, output_type=tf.int32)
    m = binary_decoding((x[:,T:]+1)/2, H)
    return t*2**H + m
    

def action_table_lookup(table, obs, H, T):

    keys = hashfunc(obs, H, T)
    actions = [table[x.numpy()] for x in keys]
    actions = tf.concat(actions, axis=0)
    return actions
    
    
    
    