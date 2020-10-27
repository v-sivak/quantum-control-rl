# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:27:01 2020

@author: Vladimir Sivak

"""

from tf_agents.drivers import dynamic_episode_driver
from gkp.utils.version_helper import TFPolicy
from gkp.gkp_tf_env import gkp_init
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
import gkp.action_script as action_scripts


class PolicyPlaceholder(TFPolicy):
    pass

class DynamicEpisodeDriverSimEnv(dynamic_episode_driver.DynamicEpisodeDriver):
    """
    This driver is a simple wrapper of the standard DynamicEpisodeDriver from 
    tf-agents. It initializes a simulated GKP environment from which data will 
    be collected according to the agent's policy.
    
    """
    def __init__(self, env_kwargs, reward_kwargs, batch_size,
                 action_script, action_scale, to_learn):
        """
        Args:
            env_kwargs (dict): optional parameters for training environment.
            reward_kwargs (dict): optional parameters for reward function.
            batch_size (int): number of episodes collected in parallel.
            action_script (str): name of action script. Action wrapper will 
                select actions from this script if they are not learned.
            action_scale (dict, str:float): dictionary mapping action dimensions
                to scaling factors. Action wrapper will rescale actions produced
                by the agent's neural net policy by these factors.
            to_learn (dict, str:bool): dictionary mapping action dimensions to 
                bool flags. Specifies if the action should be learned or scripted.
        
        """
        # Create training env and wrap it
        env = gkp_init(batch_size=batch_size, reward_kwargs=reward_kwargs,
                        **env_kwargs)
        action_script = action_scripts.__getattribute__(action_script)
        env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)        

        # create dummy placeholder policy to initialize parent class
        dummy_policy = PolicyPlaceholder(env.time_step_spec(), env.action_spec())
        
        super().__init__(env, dummy_policy, num_episodes=batch_size)        


    def run(self, episode_length=None):        
        if episode_length: 
            self.env._env.episode_length = episode_length
        super().run()

        
    def setup(self, policy, observers):
        """Setup policy and observers for the driver."""
        self._policy = policy
        self._observers = observers or []
        