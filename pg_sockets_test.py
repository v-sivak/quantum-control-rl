# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:14:26 2020

@author: qulab
"""
import socket
import time
import tensorflow as tf
import tensorflow_probability as tfp
from remote_env_tools.remote_env_tools import Client
from math import pi
from gkp.gkp_tf_env import gkp_init
import time

# def reward_sampler(a):
#     p = 1/(1+500*(a['alpha'] - 0.37)**2) # just needed a sharper reward funciton!
#     z = tfp.distributions.Bernoulli(probs=p).sample()
#     return 2*tf.cast(z, tf.float32)-1


B = 200
actions = ['alpha', 'phi_g', 'phi_e']

action_scale = {'alpha':27, 'phi_g':pi, 'phi_e':pi}


reward_kwargs = {'reward_mode' : 'measurement',
                  'sample' : True}

env = gkp_init(simulate='conditional_displacement_cal',
               reward_kwargs=reward_kwargs,
               init='vac', T=1, batch_size=B, N=50, episode_length=1,
               t_gate = 100e-9)


def reward_sampler(a):
    action = {s : a[s] * action_scale[s] for s in actions}
    time_step = env.reset()
    while not time_step.is_last()[0]:
        time_step = env.step(action)
    r = time_step.reward
    return r


# eval_reward_kwargs = {'reward_mode' : 'measurement',
#                       'sample' : False}

# env_eval = gkp_init(simulate='conditional_displacement_cal',
#                     reward_kwargs=eval_reward_kwargs,
#                     init='vac', T=1, batch_size=1, N=50, episode_length=1,
#                     t_gate = 100e-9)

# def test_eval(a):
#     action = {s : a[s] * action_scale[s] for s in actions}
#     time_step = env_eval.reset()
#     while not time_step.is_last():
#         time_step = env_eval.step(action)
#     r = time_step.reward
#     return r



client_socket = Client()
(host, port) = '127.0.0.1', 5000
client_socket.connect((host, port))

done = False
while not done:
    actions, done = client_socket.recv_data()
    if done: break
    rewards = reward_sampler(actions)
    client_socket.send_data(rewards)
    
client_socket.close()




