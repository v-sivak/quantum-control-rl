# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:27:01 2020

@author: Vladimir Sivak

"""

from tf_agents.drivers import dynamic_episode_driver
from gkp.utils.version_helper import TFPolicy
from gkp.gkp_tf_env import env_init
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
import gkp.action_script as action_scripts


class PolicyPlaceholder(TFPolicy):
    pass

class MultitaskEpisodeDriverSimEnv:
    """
    This driver allows to train the agent in multiple simulated environments.
    All environments should have the same action and observations specs, and
    the agent will use a single policy. This wrapper will create a separate
    DynamicEpisodeDriver for each environment.
    
    """
    def __init__(self, env_kwargs_list, rew_kwargs_list, batch_size,
                 action_script, action_scale, to_learn, episode_length_list,
                 env_schedule=None):
        """
        Args:
            env_kwargs_list (list[dict]): list of parameters for training 
                environment.
            reward_kwargs_list (list[dict]): list of parameters for reward 
                functions. Should correspond to 'env_kwargs_list'.
            batch_size (int): number of episodes collected in parallel.
            action_script (str): name of action script. Action wrapper will 
                select actions from this script if they are not learned.
            action_scale (dict, str:float): dictionary mapping action dimensions
                to scaling factors. Action wrapper will rescale actions produced
                by the agent's neural net policy by these factors.
            to_learn (dict, str:bool): dictionary mapping action dimensions to 
                bool flags. Specifies if the action should be learned or scripted.
            episode_length_list (list[callable: int -> int]): list of schedule 
                functions for episode durations. Schedule functions take as 
                argument int epoch number and return int episode duration for 
                this epoch. The list should correspond to 'env_kwargs_list'.
            env_schedule (callable): function mapping epoch number to index
                of the environment from the list to use during this epoch
        """
        self.env_list, self.driver_list = [], []
        self.episode_length_list = episode_length_list
        for env_kwargs, rew_kwargs in zip(env_kwargs_list, rew_kwargs_list):
            # Create training env and wrap it
            env = gkp_init(batch_size=batch_size, reward_kwargs=rew_kwargs,
                           **env_kwargs)
            action_script_m = action_scripts.__getattribute__(action_script)
            env = wrappers.ActionWrapper(env, action_script_m, action_scale, 
                                         to_learn)
    
            # create dummy placeholder policy to initialize driver
            dummy_policy = PolicyPlaceholder(
                env.time_step_spec(), env.action_spec())
            
            # create driver for this environment
            driver = dynamic_episode_driver.DynamicEpisodeDriver(
                env, dummy_policy, num_episodes=batch_size)
            
            self.env_list.append(env)
            self.driver_list.append(driver)
        
        if env_schedule is None:
            # regularly switch between environments
            self.env_schedule = lambda epoch: epoch % len(self.env_list)
        else:
            self.env_schedule = env_schedule

    def run(self, epoch):
        i = self.env_schedule(epoch)
        self.env_list[i]._env.episode_length = self.episode_length_list[i](epoch)
        self.driver_list[i].run()

    def setup(self, policy, observers):
        """Setup policy and observers for the drivers."""
        for driver in self.driver_list:
            driver._policy = policy
            driver._observers = observers or []
    
    def observation_spec(self):
        return self.env_list[0].observation_spec()
    
    def action_spec(self):
        return self.env_list[0].action_spec()
    
    def time_step_spec(self):
        return self.env_list[0].time_step_spec()