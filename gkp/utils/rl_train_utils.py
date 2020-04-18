# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:59:56 2020

@author: Vladimir Sivak
"""
import os
import numpy as np 
import h5py

def compute_avg_return(env, policy):
    """ Average return """
    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size) 
    batch_return = time_step.reward
    while not time_step.is_last()[0]:
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = env.step(action_step.action)
        batch_return += time_step.reward
    avg_return = batch_return.numpy().mean(axis=0)
    return avg_return


def save_log(log, filename, groupname):
    """
    Simple log saver.
    
    Inputs:
        log -- dictionary of arrays or lists
        filename -- name of the .hdf5 file in which to save the log
        groupname -- group name for this log
    
    """
    h5file = h5py.File(filename, 'a')
    try:
        grp = h5file.create_group(groupname)
        for name, data in log.items():
            grp.create_dataset(name, data=data)
    finally:
        h5file.close()
 