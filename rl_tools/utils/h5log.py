# Author: Ben Brock 
# Created on May 03, 2023 

import h5py
import os
import time
import numpy as np


def set_attrs(g, kwargs):
    # recursive function for storing nested dicts
    # bottom layer should be compatible with storing in an h5 group (no custom objects, etc)
    for name, value in kwargs.items():
        if isinstance(value, dict):
            sub_g = g.create_group(name)
            set_attrs(sub_g, value)
        else:
            g.attrs[name] = value

class h5log:

    def __init__(self, dir, rl_params={}):
        # dir = str, directory where h5 file will be located
        # rl_params = dict containing params for training server

        self.dir = dir
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)

        self.filename = os.path.join(self.dir,time.strftime('%Y%m%d.h5'))
        f = h5py.File(self.filename)
        if f.keys():
            keys = [k for k in f.keys() if k.isdigit()]
            group_name = str(max(map(int, keys)) + 1)
        else:
            group_name = '0'
        g = f.create_group(group_name)
        self.group_name = group_name

        rl_params['training_epochs_finished'] = 0
        rl_params['evaluation_epochs_finished'] = 0
        rl_params['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        rl_param_group = g.create_group('rl_params')
        set_attrs(rl_param_group, rl_params)

        f.close()
        
    def parse_actions(self, driver):
        # expand_dims is used to add a dimension for epoch number
        actions = {
            action_name : np.expand_dims(np.squeeze(np.array(action_history)[1:]),0)
            for action_name, action_history in driver._env.history.items()
            }
        return actions
    
    def parse_reward(self, driver):
        # expand_dims is used to add a dimension for epoch number
        reward = np.expand_dims(driver._env._episode_return.numpy(),axis=0)
        return reward

    def save_driver_data(self, driver, epoch_type):
        # saves relevant data from RL episode driver
        # (collect_driver for training epochs, eval_driver for evaluation epochs)
        # epoch_type = str, 'evaluation' or 'training'

        these_actions = self.parse_actions(driver)
        this_reward = self.parse_reward(driver)
        
        f = h5py.File(self.filename)
        g = f[self.group_name]
        g['rl_params'].attrs[epoch_type+'_epochs_finished'] += 1
        h = g.require_group(epoch_type) # creates subgroup if it doesn't exist, otherwise returns the subgroup

        if 'rewards' not in h.keys():
            h.create_dataset('rewards',
                             data = this_reward,
                             maxshape = (None,)+this_reward.shape[1:]
                             )
        else:
            h['rewards'].resize(h['rewards'].shape[0]+1,axis=0)
            h['rewards'][-1] = this_reward

        action_group = h.require_group('actions')
        for action_name, array in these_actions.items():
            if action_name not in action_group.keys():
                action_group.create_dataset(action_name,
                                            data = array,
                                            maxshape = (None,)+array.shape[1:]
                                            )
            else:
                action_group[action_name].resize(action_group[action_name].shape[0]+1,axis=0)
                action_group[action_name][-1] = array

        f.close()

    